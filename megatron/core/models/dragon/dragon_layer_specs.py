# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dragon_attention import DragonSelfAttention, DragonSelfAttentionSubmodules, DragonDiffSelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.dragon_layer import DragonLayer, DragonLayerSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.dragon_mamba_mixer import DragonMambaMixer, DragonMambaMixerSubmodules
from megatron.core.transformer.dragon_block import DragonStack, DragonStackSubmodules
try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
        TELayerNormMLP,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn('Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm

dragon_stack_spec = ModuleSpec(
    module=DragonStack,
    submodules=DragonStackSubmodules(
        dragon_layer=ModuleSpec(
            module=DragonLayer,
            submodules=DragonLayerSubmodules(
                input_layernorm=TENorm, #Would need to be removed to fuse in input_projection
                input_projection=TELayerNormColumnParallelLinear,
                self_attention=ModuleSpec(
                    module=DragonDiffSelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=DragonSelfAttentionSubmodules(
                        linear_qkv=TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        # TENorm significantly harms convergence when used
                        # for QKLayerNorm; we instead use the Apex implementation.
                        q_layernorm=FusedLayerNorm,
                        k_layernorm=FusedLayerNorm,
                    ),
                ),
                self_attn_layernorm=TENorm,
                mamba=ModuleSpec(
                    module=MambaLayer,
                    submodules=MambaLayerSubmodules(
                        mixer=ModuleSpec(
                            module=DragonMambaMixer,
                            submodules=DragonMambaMixerSubmodules(
                                in_proj=TELayerNormColumnParallelLinear
                            ),
                        ),
                    ),
                ),
                mamba_layernorm=TENorm,
                output_projection=TEColumnParallelLinear,
                
                mlp=TELayerNormMLP,
            ),
        )
    )
)



def get_dragon_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    pass
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Flag to decide the linear layer spec for MoE. Defaults to None.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    return 


