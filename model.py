import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoProcessor
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.clipseg.configuration_clipseg import CLIPSegConfig, CLIPSegTextConfig, CLIPSegVisionConfig
from transformers import AutoTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor  # Add this import at the top

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# from llava.conversation import conv_templates
# from llava.utils import disable_torch_init
# from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image
import numpy as np
# from llava.constants import IMAGE_TOKEN_INDEX
import os
import matplotlib.pyplot as plt
from transformers import BitsAndBytesConfig
from torchvision import transforms
import bitsandbytes as bnb
import torch.nn.functional as F
from transformers import GenerationConfig
import torchvision
from torchvision import models
import torch.nn.utils.spectral_norm as spectral_norm
from channel import PhysicalLayerModule
from transformers import CLIPModel, CLIPProcessor


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "CIDAS/clipseg-rd64-refined"


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->clipseg
def clipseg_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->CLIPSeg
class CLIPSegOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
class CLIPSegDecoderOutput(ModelOutput):
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            Classification scores for each pixel.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CLIPSegImageSegmentationOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        ...
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    quantization_loss: torch.FloatTensor = None
    conditional_embeddings: torch.FloatTensor = None
    pooled_output: torch.FloatTensor = None
    vision_model_output: BaseModelOutputWithPooling = None
    decoder_output: CLIPSegDecoderOutput = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["vision_model_output", "decoder_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CLIPSegVisionEmbeddings(nn.Module):
    # Copied from transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.__init__ with CLIP->CLIPSeg
    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings, height, width):
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images.
        """
        num_patches = embeddings.shape[1] - 1  # Subtract 1 for class token
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        
        # Get position embeddings without class token
        pos_emb = self.position_embedding(self.position_ids)
        class_pos_embed = pos_emb[:, :1]  # Keep class token embedding
        patch_pos_embed = pos_emb[:, 1:]  # Remove class token embedding

        # Get dimension info
        dim = embeddings.shape[-1]
        h0 = int((self.num_positions - 1) ** 0.5)  # Original grid size
        w0 = h0

        # Interpolate patch position embeddings
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(patch_height, patch_width),
            mode='bicubic',
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, patch_height * patch_width, dim)
        
        # Combine class token and patch position embeddings
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=True) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model" f" ({self.image_size}*{self.image_size})."
            )
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->CLIPSeg
class CLIPSegTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPSegTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->CLIPSeg
class CLIPSegAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->CLIPSeg
class CLIPSegMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.altclip.modeling_altclip.AltCLIPEncoderLayer with AltCLIP->CLIPSeg
class CLIPSegEncoderLayer(nn.Module):
    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPSegAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPSegMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPSegPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPSegConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, CLIPSegTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, CLIPSegVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, CLIPSegAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLIPSegMLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, CLIPSegModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


CLIPSEG_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CLIPSegConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CLIPSEG_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIPSEG_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*, defaults to `True`):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIPSEG_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*, defaults to `True`):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.altclip.modeling_altclip.AltCLIPEncoder with AltCLIP->CLIPSeg
class CLIPSegEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPSegEncoderLayer`].

    Args:
        config: CLIPSegConfig
    """

    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPSegEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class CLIPSegTextTransformer(nn.Module):
    def __init__(self, config: CLIPSegTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPSegTextEmbeddings(config)
        self.encoder = CLIPSegEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegTextConfig)
    # Adapted from transformers.models.clip.modeling_clip.CLIPTextTransformer.forward with clip->clipseg, CLIP->CLIPSeg
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIPSeg's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIPSeg/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clipseg/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIPSeg model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPSegTextModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegTextConfig

    _no_split_modules = ["CLIPSegTextEmbeddings", "CLIPSegEncoderLayer"]

    def __init__(self, config: CLIPSegTextConfig):
        super().__init__(config)
        self.text_model = CLIPSegTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPSegTextModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegTextModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPSegVisionTransformer(nn.Module):
    # Copied from transformers.models.altclip.modeling_altclip.AltCLIPVisionTransformer.__init__ with AltCLIP->CLIPSeg
    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPSegVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPSegEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor],
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPSegVisionModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPSegVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegVisionModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )


@add_start_docstrings(CLIPSEG_START_DOCSTRING)
class CLIPSegModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegConfig

    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPSegTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type CLIPSegTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPSegVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type CLIPSegVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPSegTextTransformer(text_config)
        self.vision_model = CLIPSegVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPSegTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPSegModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = True,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPSegVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    @add_start_docstrings_to_model_forward(CLIPSEG_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPSegOutput, config_class=CLIPSegConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPSegOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clipseg_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPSegOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class CLIPSegDecoderLayer(nn.Module):
    """
    CLIPSeg decoder layer, which is identical to `CLIPSegEncoderLayer`, except that normalization is applied after
    self-attention/MLP, rather than before.
    """

    # Copied from transformers.models.altclip.modeling_altclip.AltCLIPEncoderLayer.__init__ with AltCLIP->CLIPSeg
    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPSegAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPSegMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class NStepUpsample(nn.Module):
    def __init__(self, channels, n_steps):
        """
        Initializes an upsampling module that performs n_steps consecutive learnable upsampling operations.
        
        Args:
            channels (int): The number of channels in the input (and output) feature maps.
            n_steps (int): Number of upsampling steps; each step doubles the spatial resolution.
        """
        super(NStepUpsample, self).__init__()
        blocks = []
        for i in range(n_steps):
            blocks.append(
                nn.ConvTranspose2d(
                    channels, channels,
                    kernel_size=4, stride=2, padding=1, bias=False
                )
            )
            blocks.append(nn.InstanceNorm2d(channels))
            blocks.append(nn.ReLU(inplace=True))
        self.upsample = nn.Sequential(*blocks)
    
    def forward(self, x):
        
        return self.upsample(x)


class CLIPSegDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conditional_layer = config.conditional_layer
        
        # Enable gradient checkpointing to save memory
        self.gradient_checkpointing = True
        
        self.depth = len(config.extract_layers)
        

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
            # Upsample: double the spatial resolution.
            nn.ConvTranspose2d(
                config.reduce_dim, config.reduce_dim,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(config.reduce_dim),
            nn.LeakyReLU(0.2, inplace=False),
            # Refinement convolution: keeps the channel dimension.
            nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False)
            )
            for _ in range(self.depth)]
        )
        self.emb_upsample = nn.ModuleList([NStepUpsample(config.reduce_dim, n_steps=i+1) for i in range(self.depth-1)])
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.reduce_dim*2, config.reduce_dim, kernel_size=3, padding=1),
                nn.InstanceNorm2d(config.reduce_dim),  # Use ch after conv
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
                nn.InstanceNorm2d(config.reduce_dim),  # Use ch after conv
                nn.LeakyReLU(0.2, inplace=False)
            )
            for _ in range(self.depth)]
        )
        
        # Final convolution block: gradually reduce channels to 3.
        # Here we add two conv layers with non-linearities.
        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(config.reduce_dim, config.reduce_dim // 2, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(config.reduce_dim // 2, config.reduce_dim // 4, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(config.reduce_dim // 4, config.reduce_dim // 8, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(config.reduce_dim // 8, 3, kernel_size=3, padding=0)
        )
        
        decoder_config = copy.deepcopy(config.vision_config)
        decoder_config.hidden_size = config.reduce_dim
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.hidden_act = "relu"
        self.decoder_layers_global = nn.ModuleList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers)-1)])
        self.decoder_layers_local = nn.ModuleList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers)-1)])

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        local_embeddings = torch.flip(hidden_states, dims=[0])

        output = None
        for i, (local_embedding, upsample) in enumerate(zip(local_embeddings, self.upsample_blocks)):
            # Combine global and local features using gated fusion
            combined_embedding = local_embedding[:, 1:, :] #Remove CLS token
            B, L, D = combined_embedding.shape
            H = int(math.sqrt(L))
            combined_embedding = combined_embedding.reshape(B, H, H, D).permute(0, 3, 1, 2)
            if i > 0: #no need to upsample the first layer
                combined_embedding = self.emb_upsample[i-1](combined_embedding)

            if output is not None:
                output = torch.cat([output, combined_embedding], dim=1)
                output = self.conv_blocks[i-1](output)
            else:
                output = combined_embedding
        
            
            # First, upsample the new feature map.
            output = upsample(output)
            # Then add with the previous output (if exists) which should be at the same resolution.
        
        
        # Final convolution
        logits = self.final_conv(output)

        

        # Verify output size
        if logits.shape[-1] != 352:
            raise ValueError(f"Expected output size 352, but got {logits.shape[-1]}")

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_attentions] if v is not None)

        return CLIPSegDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@add_start_docstrings(
    """
    CLIPSeg model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.
    """,
    CLIPSEG_START_DOCSTRING,
)
class TextOrientedImageGeneration(nn.Module):
    def __init__(self, config, device=None):
        # Initialize parent class
        super().__init__()
        
        self.config = config
        
        torch.cuda.empty_cache()  # Clear cache before loading models
        
        # Use default CLIP with patch_size=16
        self.clip = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined", low_cpu_mem_usage=True)
        
        # Freeze CLIP model
        for param in self.clip.parameters():
            param.requires_grad = False
        self.extract_layers = config.extract_layers
        self.film_mul = nn.ModuleList([nn.Linear(config.projection_dim, config.reduce_dim) for _ in range(len(self.extract_layers))])
        self.film_add = nn.ModuleList([nn.Linear(config.projection_dim, config.reduce_dim) for _ in range(len(self.extract_layers))])
        self.reduces = nn.ModuleList(
            [nn.Linear(config.vision_config.hidden_size, config.reduce_dim) for _ in range(len(self.extract_layers))]
        )

        self.len_seg = self.config.codebook_config.embedding_dim

        self.siso = self.config.physical_config.SISO
        self.physical_layer = PhysicalLayerModule(self.config.physical_config, device=device)

        self.weight_module = WeightingModule(self.len_seg)

        self.decoder = CLIPSegDecoder(config)
        
        # Add the vector quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=self.config.codebook_config.num_embeddings,  # 64 codes in codebook
            embedding_dim=self.config.codebook_config.embedding_dim,   # 32-bit codes
            commitment_cost=self.config.codebook_config.commitment_cost,
            total_dim=config.reduce_dim,
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights"""
        # Initialize linear layers
        for module in [self.film_mul, self.film_add, self.reduces]:
            if isinstance(module, nn.Linear): 
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def get_conditional_embeddings(
        self,
        batch_size: int = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        conditional_pixel_values: Optional[torch.Tensor] = None,
    ):
        if input_ids is not None:
            # compute conditional embeddings from texts
            if len(input_ids) != batch_size:
                raise ValueError("Make sure to pass as many prompt texts as there are query images")
            with torch.no_grad():
                conditional_embeddings = self.clip.get_text_features(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids
                )
        elif conditional_pixel_values is not None:
            # compute conditional embeddings from images
            if len(conditional_pixel_values) != batch_size:
                raise ValueError("Make sure to pass as many prompt images as there are query images")
            with torch.no_grad():
                conditional_embeddings = self.clip.get_image_features(conditional_pixel_values)
        else:
            raise ValueError(
                "Invalid conditional, should be either provided as `input_ids` or `conditional_pixel_values`"
            )

        return conditional_embeddings

    @add_start_docstrings_to_model_forward(CLIPSEG_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPSegImageSegmentationOutput, config_class=CLIPSegTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        conditional_pixel_values: Optional[torch.FloatTensor] = None,
        conditional_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = True,
        return_dict: Optional[bool] = None,
        stage: Optional[int] = 0,
        textalign: Optional[bool] = True,
    ) -> Union[Tuple, CLIPSegOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, CLIPSegForImageSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["a cat", "a remote", "a blanket"]
        >>> inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> print(logits.shape)
        torch.Size([3, 352, 352])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        

        # step 1: forward the query images through the frozen CLIP vision encoder
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=True,  # we need the intermediate hidden states
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            pooled_output = self.clip.visual_projection(vision_outputs[1])

            hidden_states = vision_outputs.hidden_states if return_dict else vision_outputs[2]
            # we add +1 here as the hidden states also include the initial embeddings
            activations = [hidden_states[i + 1] for i in self.extract_layers]

            # update vision_outputs
            if return_dict:
                vision_outputs = BaseModelOutputWithPooling(
                    last_hidden_state=vision_outputs.last_hidden_state,
                    pooler_output=vision_outputs.pooler_output,
                    hidden_states=vision_outputs.hidden_states if output_hidden_states else None,
                    attentions=vision_outputs.attentions,
                )
            else:
                vision_outputs = (
                    vision_outputs[:2] + vision_outputs[3:] if not output_hidden_states else vision_outputs
                )

        # step 2: compute conditional embeddings
        if conditional_embeddings is None:
            conditional_embeddings = self.get_conditional_embeddings(
                batch_size=pixel_values.shape[0],
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                conditional_pixel_values=conditional_pixel_values,
            )
        else:
            if conditional_embeddings.shape[0] != pixel_values.shape[0]:
                raise ValueError(
                    "Make sure to pass as many conditional embeddings as there are query images in the batch"
                )
            if conditional_embeddings.shape[1] != self.config.projection_dim:
                raise ValueError(
                    "Make sure that the feature dimension of the conditional embeddings matches"
                    " `config.projection_dim`."
                )

        local_embeddings = []
        transmitted_indices = []
        quantized_activations = []
        quantization_loss = 0
        
        for i, (activation, reduce) in enumerate(zip(activations, self.reduces)):
            activation = reduce(activation)
            
            if textalign:
                local_embedding = self.film_mul[i](conditional_embeddings) * activation.permute(1, 0, 2) + self.film_add[i](conditional_embeddings)
            else:
                local_embedding = activation.permute(1, 0, 2)
            local_embeddings.append(local_embedding.permute(1, 0, 2))
            quantized_activation, q_loss, quantized_indices = self.quantizer(local_embedding.permute(1, 0, 2))
            quantized_activations.append(quantized_activation)
            quantization_loss += q_loss
            # Use the indices for simulating the physical layer (non-differentiable)
            transmitted_index = quantized_indices.detach()  # simulate transmission
            transmitted_indices.append(transmitted_index)
        
        local_embeddings = torch.stack(local_embeddings, dim=0)
        quantized_activations = torch.stack(quantized_activations, dim=0)
        if stage == 1:
            transmitted_indices = torch.stack(transmitted_indices, dim=0)
            received = transmitted_indices
        else:
            if self.siso:
                received = self.physical_layer(transmitted_indices)
            else:
                importance_weight = self.weight_module(local_embeddings, self.len_seg)
                received = self.physical_layer(transmitted_indices, importance_weight)  #  [B, len, group] recovered indices
        recovered_embedding = self.quantizer.recover_embeddings(received)
        
        # STE: replace hard recovered with soft quantized during backward
        mixed_embedding = quantized_activations + (recovered_embedding - quantized_activations).detach()

        # step 3: forward through decoder
        if self.training:
            decoder_outputs = self.decoder(
                mixed_embedding,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            decoder_outputs = self.decoder(
                recovered_embedding,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]

        loss = None
        if labels is not None:
            # move labels to the correct device to enable PP
            labels = labels.to(logits.device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        if not return_dict:
            output = (logits, conditional_embeddings, pooled_output, vision_outputs, decoder_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPSegImageSegmentationOutput(
            loss=loss,
            logits=logits,
            quantization_loss=quantization_loss,
            conditional_embeddings=conditional_embeddings,
            pooled_output=pooled_output,
            vision_model_output=vision_outputs,
            decoder_output=decoder_outputs,
        )


class AnswerGenerationModel(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch16', device=None):
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load CLIP model and processor
        self.clip = CLIPModel.from_pretrained(model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Perceptual loss
        self.perc_loss = nn.L1Loss()

        for param in self.clip.parameters():
            param.requires_grad = False  # Freeze CLIP parameters


    def forward(self, images, generated_images, questions, answers):
        """Forward pass using CLIP for perceptual alignment"""
        # Ensure images and texts are lists
        if not isinstance(questions, list):
            questions = [questions]
        if not isinstance(answers, list):
            answers = [answers]

        prompts = [f"Q: {q} A: {a}" for q, a in zip(questions, answers)]
        images = images.detach().clamp(0, 1).float()
        generated_images = generated_images.detach().clamp(0, 1).float()

        # Tokenize with text
        inputs_real = self.processor(text=prompts, images=images, return_tensors="pt", padding=True, do_rescale=False)
        inputs_gen = self.processor(text=prompts, images=generated_images, return_tensors="pt", padding=True, do_rescale=False)

        # Move inputs to device
        inputs_real = {k: v.to(self.device) for k, v in inputs_real.items()}
        inputs_gen = {k: v.to(self.device) for k, v in inputs_gen.items()}
        with torch.no_grad():
            img_feats_real = self.clip.get_image_features(pixel_values=inputs_real["pixel_values"])
            img_feats_gen  = self.clip.get_image_features(pixel_values=inputs_gen["pixel_values"])
            txt_feats      = self.clip.get_text_features(input_ids=inputs_real["input_ids"], attention_mask=inputs_real["attention_mask"])

        img_feats_real = F.normalize(img_feats_real, dim=-1, eps=1e-6)
        img_feats_gen  = F.normalize(img_feats_gen, dim=-1, eps=1e-6)
        txt_feats      = F.normalize(txt_feats, dim=-1, eps=1e-6)

        # Compute cosine similarity
        cos_sim_real = (img_feats_real * txt_feats).sum(dim=-1)
        cos_sim_gen  = (img_feats_gen * txt_feats).sum(dim=-1)
        

        loss_perc = F.l1_loss(cos_sim_gen, cos_sim_real)*10

        return {
            'cos_gen_features': cos_sim_gen,
            'cos_target_features': cos_sim_real,
            'loss_perc': loss_perc,
        }



class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=["relu1_2", "relu2_2", "relu3_3"], use_l1=True):
        super(VGGPerceptualLoss, self).__init__()
    
        # Load VGG-16 and extract its feature layers
        vgg = models.vgg16(pretrained=True).features
        
        # Convert all ReLUs to non-inplace version
        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
    
        # Define layer indices for feature extraction
        self.layer_map = {
            "relu1_2": 3,   # relu1_2
            "relu2_2": 8,   # relu2_2
            "relu3_3": 15,  # relu3_3
        }
    
        self.selected_layers = [self.layer_map[l] for l in layers]
    
        # Extract required layers in a single pass
        self.vgg = nn.Sequential(*list(vgg.children())[: max(self.selected_layers) + 1])
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters
    
        # Choose loss function (L1 is better)
        self.loss_fn = F.l1_loss if use_l1 else F.mse_loss
    
    def forward(self, x, y):
        loss = 0
        
        for i, layer in enumerate(self.vgg.children()):
            if i in self.selected_layers:
                # Process x and y separately
                x_feat = layer(x)
                y_feat = layer(y)
                loss += self.loss_fn(x_feat, y_feat)
                
        return loss


class VQAWithSegmentation(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        self.device = device
        self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", do_rescale=False)
        
        # Initialize VGG perceptual loss
        self.vgg_loss = VGGPerceptualLoss(
            layers=["relu1_2", "relu2_2", "relu3_3"],
            use_l1=True
        ).to(device)
        
        # Initialize models
        self.image_generation_model = TextOrientedImageGeneration(config=config, device=self.device)
        self.perceptual_model = AnswerGenerationModel(
            model_name='openai/clip-vit-base-patch16',
            device=self.device
        )

        # Reconstruction Loss (e.g., L1 Loss)
        self.recon_loss = nn.L1Loss()
        # Define a transform to match VGG's expected input for perceptual loss
        self.perceptual_preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        
    def forward(self, images, questions, answer, stage=None, textalign=True):
        # Process images and text separately
        image_inputs = self.processor.image_processor(
            images=images,
            return_tensors="pt"
        )
        
        # Process text with padding and truncation
        text_inputs = self.processor.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP's max context length
            return_tensors="pt"
        )
        
        # Combine inputs
        inputs = {
            **image_inputs,
            **text_inputs
        }
        
        # Move all inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = self.image_generation_model(**inputs, stage=stage, textalign=textalign)
        generated_images = outputs.logits  # Keep batch dimension
        quantization_loss = outputs.quantization_loss
        
        # --------------------
        # 1. Reconstruction Loss
        # --------------------
        recon_loss = self.recon_loss(generated_images, images)
        
        # --------------------
        # 2. Perceptual Loss
        # --------------------
        # Add batch dimension for interpolation
        perceptual_loss = self.perceptual_model(images, generated_images, questions, answer)['loss_perc']
        
        # Add VGG perceptual loss
        vgg_loss = self.vgg_loss(generated_images, images)
        
        return {
            'generated_images': outputs.logits,
            'loss_recon': recon_loss,
            'loss_perc': perceptual_loss,
            'loss_vgg': vgg_loss,
            'quantization_loss': quantization_loss,
        }

__all__ = [
    "CLIPSegModel",
    "CLIPSegPreTrainedModel",
    "CLIPSegTextModel",
    "CLIPSegVisionModel",
    "CLIPSegForImageSegmentation",
]


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        """
        A PatchGAN discriminator that classifies patches in the input image.
        
        Args:
            in_channels (int): Number of channels in the input image (typically 3 for RGB).
            base_channels (int): Base number of channels for the discriminator.
        """
        super().__init__()
        self.model = nn.Sequential(
            # First layer: no normalization, stride=2 reduces resolution.
            spectral_norm(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second layer: increase channels, stride=2.
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third layer: increase channels, stride=2.
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth layer: use stride=1 to capture local details.
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final output layer: output a 1-channel prediction map.
            spectral_norm(nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1))
        )
        self.apply(self._init_weights)  # Apply weight initialization
        
    def forward(self, x):
        # Add input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B,C,H,W), got shape {x.shape}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
        
        # Ensure float type
        x = x.float()
        
        return self.model(x)
    
    def get_intermediate_features(self, x):
        """Get intermediate feature maps for feature matching loss"""
        features = []
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        return features
    
    def _init_weights(self, m):
        """Weight initialization for Conv2d layers"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)




class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=32, embedding_dim=32, commitment_cost=0.25, total_dim=512, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.num_groups = total_dim // embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        
        # Create separate embedding tables for each group
        self.embeds = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for _ in range(self.num_groups)
        ])
        
        # Initialize embedding weights
        for embed in self.embeds:
            embed.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        # Input shape: [..., total_dim]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, input_shape[-1])
        
        # Split input into groups
        grouped_inputs = flat_input.view(-1, self.num_groups, self.embedding_dim)
        
        total_loss = 0
        quantized_list = []
        indices_list = []
        
        # Process each group separately
        for i, embed in enumerate(self.embeds):
            group_input = grouped_inputs[:, i, :]

            group_input = group_input / (group_input.norm(dim=1, keepdim=True) + 1e-6)
            
            # Calculate distances for this group
            distances = (torch.sum(group_input**2, dim=1, keepdim=True) 
                        + torch.sum(embed.weight**2, dim=1)
                        - 2 * torch.matmul(group_input, embed.weight.t()))
            
            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).long()  # shape: [N]
            quantized = F.embedding(encoding_indices, embed.weight)  # shape: [N, embedding_dim]

            
            # Loss for this group
            e_latent_loss = torch.mean((quantized.detach() - group_input)**2)
            q_latent_loss = torch.mean((quantized - group_input.detach())**2)
            group_loss = q_latent_loss + self.commitment_cost * e_latent_loss
            
            total_loss += group_loss
            
            # Straight through estimator
            quantized = group_input + (quantized - group_input).detach()
            
            
            quantized_list.append(quantized)
            indices_list.append(encoding_indices.unsqueeze(1))

        
        # Combine quantized vectors and reshape
        quantized = torch.stack(quantized_list, dim=1)
        quantized = quantized.view(*input_shape)
        
        # Stack all indices
        encoding_indices = torch.cat(indices_list, dim=1)
        encoding_indices = encoding_indices.view(*input_shape[:-1], self.num_groups)
        
        # Average loss across groups
        total_loss = total_loss / self.num_groups
        
        return quantized, total_loss, encoding_indices

    @staticmethod
    def indices_to_embedding_ste(indices, embed_table):
        indices = indices.long()
        embedding = F.embedding(indices, embed_table)
        return embedding + (embedding - embedding).detach()

    def recover_embeddings(self, quantized_indices):
        """
        Given quantized indices (e.g. of shape [B, num_groups] or [B, seq_len, num_groups]),
        recover the continuous embeddings using an embedding lookup with a straight-through estimator.
        """
        # Assume quantized_indices shape is [B, num_groups]
        # (If it's [B, seq_len, num_groups], adjust accordingly.)
        orig_shape = quantized_indices.shape[:-1]  # [...], excluding group dim
        num_groups = quantized_indices.shape[-1]
        flat_indices = quantized_indices.view(-1, num_groups)  # [N, num_groups]

        recovered_list = []
        for i, embed in enumerate(self.embeds):
            indices = flat_indices[:, i]  # [N]
            recovered = self.indices_to_embedding_ste(indices, embed.weight)  # [N, embedding_dim]
            recovered_list.append(recovered)

        recovered_flat = torch.cat(recovered_list, dim=-1)  # [N, total_dim]
        recovered = recovered_flat.view(*orig_shape, -1)    # [..., total_dim]
        return recovered
    


class WeightingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        Args:
            input_dim (int): Dimension of the token embedding (should equal seg).
            hidden_dim (int): Hidden size of the network.
        """
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs importance weights in (0, 1)
        )

    def forward(self, embeddings, seg):
        """
        Args:
            embeddings: Tensor of shape [L, B, len, D], where
                        L = number of layers,
                        B = batch size,
                        len (N) = sequence length,
                        D = token dimension.
            seg: An integer factor such that D is divisible by seg.
        Returns:
            importance_weights: Tensor of shape [L, B, len]
                                (one importance weight per layer, per sample, per token position)
        """
        L, B, N, D = embeddings.shape
        assert D % seg == 0, "D must be divisible by seg."

        # Step 1. Permute to bring batch and sequence dimensions together:
        # From [L, B, N, D] to [B, N, L, D]
        x = embeddings.permute(1, 2, 0, 3)  # shape: [B, N, L, D]

        # Step 2. Reshape D into two factors: (D // seg) and seg.
        x = x.view(B, N, L, D // seg, seg)  # shape: [B, N, L, D//seg, seg]

        # Step 3. Combine the L and (D // seg) dimensions:
        x = x.reshape(B, N, L * (D // seg), seg)  # shape: [B, N, L*(D//seg), seg]

        # Step 4. Merge B and N so that we can apply the policy network on each token of size seg.
        x_flat = x.view(B * N, L * (D // seg), seg)  # shape: [B*N, L*(D//seg), seg]

        # Apply the policy network on each token.
        # We flatten the last two dims so that each row is a token of dimension 'seg'
        tokens = x_flat.contiguous()               # shape: [B*N, L*(D//seg), seg]
        weights = self.policy(tokens)              # shape: [B*N, L*(D//seg), 1]
        # Reshape back to [B*N, L*(D//seg)]
        weights = weights.squeeze(-1)

        # Normalize each row so that its sum equals 1.
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

