from dataclasses import dataclass
from enum import auto
from typing import Optional

from elias.config import Config, StringEnum


class HeadTransformerType(StringEnum):
    MESH_TOKENS = auto()
    UV_TEXTURE = auto()


class CrossAttentionType(StringEnum):
    Q2K = auto()
    Q2QK = auto()
    QK2QK = auto()


@dataclass
class TransformerConfig(Config):
    n_layers: int
    d_hidden: int
    n_heads: int
    use_custom_attention: bool = False
    use_qk_norm: bool = False
    use_layer_norm_keys: bool = False
    use_alternating_self_attention: bool = False
    use_causal_attention: bool = True


@dataclass
class HeadTransformerConfig(Config):
    transformer: TransformerConfig
    res_head_tokens: int
    head_transformer_type: HeadTransformerType = HeadTransformerType.MESH_TOKENS
    cross_attention_type: CrossAttentionType = CrossAttentionType.Q2K
    use_lam_transformer: bool = False
    use_lam_point_embedder: bool = False
    res_image_tokens: Optional[int] = None
    n_input_views: int = 1
    block_size_estimate_version: int = 1
    d_expression_codes: Optional[int] = None
    n_expression_tokens: Optional[int] = 4
    use_head_tokens: bool = True
    n_layers_expression_transformer: int = 4
    use_dataset_ids: bool = False
    head_template: str = 'gghead_template'
