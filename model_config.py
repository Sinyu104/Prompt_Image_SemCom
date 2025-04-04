from dataclasses import dataclass
from typing import List
import json

@dataclass
class VisionConfig:
    attention_dropout: float
    dropout: float
    hidden_act: str
    hidden_size: int
    image_size: int
    initializer_factor: float
    initializer_range: float
    intermediate_size: int
    layer_norm_eps: float
    model_type: str
    num_attention_heads: int
    num_channels: int
    num_hidden_layers: int
    patch_size: int
    transformers_version: str

    def __post_init__(self):
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")

@dataclass
class ModelConfig:
    llava_model_path: str
    clipseg_model_path: str
    vision_tower_path: str
    image_size: int
    patch_size: int
    num_channels: int
    hidden_size: int
    projection_dim: int
    reduce_dim: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    use_complex_transposed_convolution: bool
    conditional_layer: int
    decoder_attention_dropout: float
    decoder_hidden_act: str
    decoder_intermediate_size: int
    decoder_num_attention_heads: int
    extract_layers: List[int]
    vision_config: VisionConfig
    gradient_checkpointing: bool
    freeze_vision_tower: bool
    freeze_llm: bool
    use_8bit_quant: bool
    low_cpu_mem_usage: bool
    use_return_dict: bool

    @classmethod
    def from_json(cls, json_path: str) -> 'ModelConfig':
        """
        Load configuration from a JSON file and return a ModelConfig instance.
        """
        with open(json_path, 'r') as f:
            config_data = json.load(f)

        # Extract vision_config data and create VisionConfig instance
        vision_cfg_data = config_data.pop('vision_config')
        vision_cfg = VisionConfig(**vision_cfg_data)

        # Create ModelConfig instance with the remaining config data
        return cls(vision_config=vision_cfg, **config_data)         