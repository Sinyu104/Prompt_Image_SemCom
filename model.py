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
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX
import os
import matplotlib.pyplot as plt
from transformers import BitsAndBytesConfig
from torchvision import transforms
import bitsandbytes as bnb
import torch.nn.functional as F
from transformers import GenerationConfig


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


class CLIPSegDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conditional_layer = config.conditional_layer
        if config.use_complex_transposed_convolution:
            # Calculate upscale factor from 22x22 to 352x352
            input_size = 22  # From CLIP's 16x16 patches on 352x352 image
            target_size = 352
            upscale_factor = target_size // input_size  # 16
            
            self.transposed_convolution = nn.Sequential(
                # Initial processing at 22x22
                nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding='same'),
                nn.BatchNorm2d(config.reduce_dim),
                nn.ReLU(),
                
                # First upsampling: 22x22 -> 44x44
                nn.ConvTranspose2d(
                    config.reduce_dim,
                    config.reduce_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0
                ),
                nn.BatchNorm2d(config.reduce_dim // 2),
                nn.ReLU(),
                
                # Second upsampling: 44x44 -> 88x88
                nn.ConvTranspose2d(
                    config.reduce_dim // 2,
                    config.reduce_dim // 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0
                ),
                nn.BatchNorm2d(config.reduce_dim // 4),
                nn.ReLU(),
                
                # Third upsampling: 88x88 -> 176x176
                nn.ConvTranspose2d(
                    config.reduce_dim // 4,
                    config.reduce_dim // 8,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0
                ),
                nn.BatchNorm2d(config.reduce_dim // 8),
                nn.ReLU(),
                
                # Final upsampling: 176x176 -> 352x352
                nn.ConvTranspose2d(
                    config.reduce_dim // 8,
                    3,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0
                ),
                # Edge smoothing
                nn.ReflectionPad2d(1),
                nn.Conv2d(3, 3, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(3, 3, kernel_size=3, padding=0),
                nn.Sigmoid()
            )
        else:
            self.transposed_convolution = nn.ConvTranspose2d(
                config.reduce_dim, 3, config.vision_config.patch_size, stride=config.vision_config.patch_size
            )
            
        # Initialize weights properly based on module type
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                # Use smaller standard deviation for more centered outputs
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                
                # Special handling for final conv layer before sigmoid
                if isinstance(m, nn.Conv2d) and m.out_channels == 3:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        
        # Apply initialization
        self.apply(init_weights)

        depth = len(config.extract_layers)
        self.reduces = nn.ModuleList(
            [nn.Linear(config.vision_config.hidden_size, config.reduce_dim) for _ in range(depth)]
        )

        decoder_config = copy.deepcopy(config.vision_config)
        decoder_config.hidden_size = config.reduce_dim
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.hidden_act = "relu"
        self.layers = nn.ModuleList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers))])

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        local_embeddings: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        global_embeddings = hidden_states[::-1]
        local_embeddings = local_embeddings[::-1]
        

        output = None
        for i, (global_embedding, local_embedding, layer) in enumerate(zip(global_embeddings, local_embeddings, self.layers)):
            
            # Scale global features before combining
            combined_embedding = local_embedding
            
            layer_outputs = layer(
                    combined_embedding,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=output_attentions
                )

            if output is not None:
                output = layer_outputs[0] + output
            else:
                output = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states += (output,)

            if output_attentions:
                all_attentions += (layer_outputs[1],)
            
        
        output = output[:, 1:, :].permute(0, 2, 1)
        size = int(math.sqrt(output.shape[2]))  # Should be 22
        batch_size = output.shape[0]
        output = output.view(batch_size, output.shape[1], size, size)
        
        # Add activation to help with feature distribution
        output = F.gelu(output)
        
        # Let transposed convolution handle the upsampling
        logits = self.transposed_convolution(output)
        
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
        
        self.decoder = CLIPSegDecoder(config)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights"""
        # Initialize linear layers
        for module_list in [self.film_mul, self.film_add, self.reduces]:
            for module in module_list:
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
        for i, (activation, reduce) in enumerate(zip(activations, self.reduces)):
            activation = reduce(activation)
            local_embedding = self.film_mul[i](conditional_embeddings) * activation.permute(1, 0, 2) + self.film_add[i](conditional_embeddings)
            local_embeddings.append(local_embedding.permute(1, 0, 2))



        # step 3: forward through decoder
        decoder_outputs = self.decoder(
            activations,
            local_embeddings,
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
            conditional_embeddings=conditional_embeddings,
            pooled_output=pooled_output,
            vision_model_output=vision_outputs,
            decoder_output=decoder_outputs,
        )

    def visualize_outputs(self, images, questions, outputs, save_dir="visualizations"):
        """Visualize model outputs including original, mask, masked image and answer"""
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        batch_size = images.shape[0]
        for i in range(batch_size):
            # Create subplot
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle(f'Question: {questions[i]}\nAnswer: {outputs["answers"][i]}', wrap=True)
            
            # Original image
            img = images[i].cpu().numpy()
            if img.shape[0] == 3:  # If channels first
                img = np.transpose(img, (1, 2, 0))
            axs[0, 0].imshow(img)
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')
            
            # Segmentation mask
            mask = outputs['segmentation_masks'][i].cpu().numpy()
            axs[0, 1].imshow(mask, cmap='jet')
            axs[0, 1].set_title('Segmentation Mask')
            axs[0, 1].axis('off')
            
            # Masked image
            masked_img = outputs['masked_images'][i].cpu().numpy()
            if masked_img.shape[0] == 3:
                masked_img = np.transpose(masked_img, (1, 2, 0))
            axs[1, 0].imshow(masked_img)
            axs[1, 0].set_title('Masked Image')
            axs[1, 0].axis('off')
            
            # Overlay mask on original
            overlay = img.copy()
            mask_rgb = np.stack([mask]*3, axis=2)
            overlay = overlay * 0.7 + mask_rgb * 0.3
            axs[1, 1].imshow(overlay)
            axs[1, 1].set_title('Overlay')
            axs[1, 1].axis('off')
            
            # Save plot
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'visualization_{timestamp}_{i}.png')
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved visualization to {save_path}")


class AnswerGenerationModel(nn.Module):
    def __init__(self, model_path, load_in_8bit=False, device=None):
        super().__init__()
        
        # Initialize LLaVA model
        self.llava = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=None,  # Let DDP handle device placement
        )
        if not self.llava.get_vision_tower().is_loaded:
            self.llava.get_vision_tower().load_model()
        
        # Freeze LLaVA model
        for param in self.llava.parameters():
            param.requires_grad = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        # Configure generation settings
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.llava.generation_config = self.generation_config
        
        # Perceptual Loss (e.g., MSE Loss)
        self.perc_loss = nn.MSELoss()
        
    def forward(self, images, generated_images, questions):
        """Forward pass for answer generation"""
        # Process all questions in batch at once
        if isinstance(questions, list):
            messages = [{"role": "user", "content": f'<image>\n{q}'} for q in questions]
        else:
            messages = [{"role": "user", "content": f'<image>\n{q}'} for q in questions]
        
        # Batch process all texts
        texts = [
            self.tokenizer.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            ) for msg in messages
        ]
        
        # Process all inputs in batch
        input_ids_list = []
        attention_mask_list = []
        for text in texts:
            chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
            input_ids = torch.tensor(chunks[0] + [-200] + chunks[1], dtype=torch.long)
            input_ids_list.append(input_ids)
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask_list.append(torch.ones_like(input_ids))
        
        # Pad input_ids to same length
        max_len = max(ids.size(0) for ids in input_ids_list)
        # Ensure all ranks use the same max_len
        if torch.distributed.is_initialized():
            max_len_tensor = torch.tensor(max_len, device=self.llava.device)
            torch.distributed.all_reduce(max_len_tensor, op=torch.distributed.ReduceOp.MAX)
            max_len = int(max_len_tensor.item())
        
        # Pad both input_ids and attention_mask
        padded_ids = torch.stack([
            F.pad(ids, (0, max_len - ids.size(0)), value=self.tokenizer.pad_token_id)
            for ids in input_ids_list
        ]).to(self.llava.device)
        
        attention_mask = torch.stack([
            F.pad(mask, (0, max_len - mask.size(0)), value=0)
            for mask in attention_mask_list
        ]).to(self.llava.device)
        
        # Process entire batch at once for feature extraction
        with torch.no_grad():
            image_output = self.llava(
                padded_ids,
                images=images,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True
            ).hidden_states[-1]
            
            gen_output = self.llava(
                padded_ids,
                images=generated_images,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True
            ).hidden_states[-1]
        
        loss_perc = self.perc_loss(gen_output, image_output)
        
        # During training, skip text generation completely
        responses = [""] * images.size(0)
        
        # Only do generation during eval
        if not self.training:
            with torch.no_grad():
                # Generate text only on rank 0
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    for i in range(images.size(0)):
                        output_id = self.llava.generate(
                            padded_ids[i:i+1],
                            images=generated_images[i:i+1],
                            attention_mask=attention_mask[i:i+1],
                            max_new_tokens=2048,
                            use_cache=True,
                            synced_gpus=False,  # Disable syncing during generation
                            generation_config=self.generation_config
                        )[0]
                        responses[i] = self.tokenizer.decode(
                            output_id[padded_ids[i].size(0):],
                            skip_special_tokens=True
                        ).strip()
                # Broadcast responses from rank 0 to all other ranks
                if torch.distributed.is_initialized():
                    responses_tensor = torch.tensor([len(r) for r in responses], device=self.llava.device)
                    torch.distributed.broadcast(responses_tensor, src=0)
                    for i, length in enumerate(responses_tensor):
                        if torch.distributed.get_rank() != 0:
                            responses[i] = " " * length  # Placeholder for non-rank-0
        
        return {
            'text': responses,
            'gen_features': gen_output,
            'target_features': image_output,
            'loss_perc': loss_perc,
        }


class VQAWithSegmentation(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        self.device = device
        self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", do_rescale=False)
        # Initialize models
        self.image_generation_model = TextOrientedImageGeneration(config=config, device=self.device)
        self.perceptual_model = AnswerGenerationModel(
            model_path="qnguyen3/nanoLLaVA-1.5",
            load_in_8bit=False,
            device=self.device
        )
        # Reconstruction Loss (e.g., L1 Loss)
        self.recon_loss = nn.L1Loss()
        # Define a transform to match VGG's expected input for perceptual loss
        self.perceptual_preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
        
        
        
    def forward(self, images, questions):
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
        outputs = self.image_generation_model(**inputs)
        generated_images = outputs.logits  # Keep batch dimension
        
        # --------------------
        # 1. Reconstruction Loss
        # --------------------
        recon_loss = self.recon_loss(generated_images, images)
        
        # --------------------
        # 2. Perceptual Loss
        # --------------------
        # Add batch dimension for interpolation
        generated_normalized = F.interpolate(self.perceptual_preprocess(generated_images), 
                                          size=(384, 384), mode='bilinear', align_corners=False)
        targets_normalized = F.interpolate(self.perceptual_preprocess(images),
                                         size=(384, 384), mode='bilinear', align_corners=False)

        perceptual_loss = self.perceptual_model(targets_normalized, generated_normalized, questions)['loss_perc']
        
        return {
            'generated_images': outputs.logits,
            'loss_recon': recon_loss,
            'loss_perc': perceptual_loss
        }


__all__ = [
    "CLIPSegModel",
    "CLIPSegPreTrainedModel",
    "CLIPSegTextModel",
    "CLIPSegVisionModel",
    "CLIPSegForImageSegmentation",
]


def tensor_to_pil(tensor, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), rescale_factor=1/255, data_format="channels_first"):
    """
    Converts a PyTorch tensor back to a PIL Image by reversing preprocessing transformations.

    Args:
        tensor (torch.Tensor): The preprocessed image tensor.
        image_mean (tuple): Mean values used for normalization.
        image_std (tuple): Standard deviation values used for normalization.
        rescale_factor (float): Factor used to rescale images during preprocessing.
        data_format (str): The format of the channel dimension, either "channels_first" (CHW) or "channels_last" (HWC).

    Returns:
        PIL.Image: The restored PIL image.
    """

    # Remove batch dimension if present (from shape [1, 3, H, W] to [3, H, W])
    if tensor.ndim == 4:  
        tensor = tensor.squeeze(0)

    # Ensure the tensor is in NumPy format
    array = tensor.detach().cpu().numpy()

    # Reverse channel formatting: (C, H, W) -> (H, W, C) if needed
    if data_format == "channels_first":
        array = np.transpose(array, (1, 2, 0))  # Convert from CHW to HWC

    # Reverse normalization: (x * std) + mean
    array = (array * np.array(image_std)) + np.array(image_mean)

    # Reverse rescaling: Multiply by 1/rescale_factor (i.e., 255 if rescale_factor = 1/255)
    array = array / rescale_factor

    # Clip values to ensure they are within valid image range [0, 255]
    array = np.clip(array, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    return Image.fromarray(array)



