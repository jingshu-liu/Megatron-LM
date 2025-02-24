# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Union

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.dragon_config import DragonConfig
from megatron.core.utils import make_viewless_tensor


@dataclass
class DragonLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    input_projection: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_layernorm: Union[ModuleSpec, type] = IdentityOp

    mamba_layernorm: Union[ModuleSpec, type] = IdentityOp
    mamba: Union[ModuleSpec, type] = IdentityOp
    
    output_projection: Union[ModuleSpec, type] = IdentityOp

    mlp: Union[ModuleSpec, type] = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class CacheSharing(Enum):
    FIRST = 'first'
    SECOND = 'second'


class BaseFragonLayer(ABC):
    """A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass


class DragonLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: DragonConfig,
        submodules: DragonLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        cache_sharing: CacheSharing = CacheSharing.FIRST,
        window_size: Optional[tuple[int, int]] = None,
        mamba_ssm_ngroups: int = 8,
        
    ):
        super().__init__(config=config)
        
        self.submodules_config = submodules
        self.layer_number = layer_number + self._get_layer_offset()
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.cache_sharing = cache_sharing

        # [Module 1: Input Layernorm] 
        # Input layernorm will be fused with the input projection
        # But it's work in progress
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        
        
    
        # [Module 2: InputProjection]
        intermediate_size = int(2 * self.config.hidden_size) # Mamba Expand
        # attention_head_size = int(intermediate_size / self.config.num_attention_heads)
        
        #Save params for forward pass
        self.intermediate_size = intermediate_size
        # self.attention_head_size = attention_head_size
        # self.num_attention_heads = self.config.num_attention_heads
        # self.num_query_groups = self.config.num_query_groups
        
        #To implement
        # self.layer_norm_scaling = self.config.layer_norm_scaling
        
        
        
        # input_proj_size = intermediate_size #For mamba branch
        # input_proj_size += intermediate_size #For gate in Mamba Branch
        # input_proj_size += intermediate_size #For Query in Attention Branch
        # if cache_sharing == CacheSharing.FIRST:
        #     input_proj_size += attention_head_size * self.config.num_query_groups * 2 # will go to attn branch - key, value
          
        """
        self.input_proj = build_module(
            submodules.input_projection,
            self.config.hidden_size,
            input_proj_size,        
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            layer_number=layer_number,
            bias=True,
            skip_bias_add=True,
            normalization="RMSNorm",
            tp_comm_buffer_name='input_proj',
            return_layernorm_output=False,
        )
        """
        
        # [Module 3: SelfAttention Branch]
        # Need to handle Query Key Cache in this module
                         
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config, 
            layer_number=layer_number,
            window_size=window_size,
        )

        # [Module 4: Post SelfAttention Norm]
        self.self_attn_layernorm = build_module(
            submodules.self_attn_layernorm,
            config=self.config,
            hidden_size=self.intermediate_size,
            eps=self.config.layernorm_epsilon,
        )
        
        # [Module 5: Mamba Branch]
        self.mamba_mixer = build_module(
            submodules.mamba,
            self.config,
            mamba_ssm_ngroups=mamba_ssm_ngroups,
            layer_number=layer_number,
        )
        
        # [Module 6: Post Mamba Norm]
        self.mamba_layernorm = build_module(
            submodules.mamba_layernorm,
            config=self.config,
            hidden_size=self.intermediate_size,
            eps=self.config.layernorm_epsilon,
        )

        
        # [Module 7: Output Proj] Optional Layernorm before MLP
        
        self.output_projection = build_module(
            submodules.output_projection,
            self.intermediate_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=True,
            tp_comm_buffer_name='output_proj',
            is_expert=False,
        )

        # [Module 8: MLP block]
        self.mlp = build_module(
            submodules.mlp, 
            config=self.config,
            init_method=self.config.init_method,
            input_size=self.config.hidden_size,
            hidden_size=4*self.config.hidden_size,
            bias=True,
            skip_bias_add=False            
        )
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        
    def _get_layer_offset(self):
        """Get the index number of this layer, given the level of pipelining."""
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        num_layers_per_pipeline_rank = (
            self.config.num_layers // self.config.pipeline_model_parallel_size
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                if (
                    self.config.first_pipeline_num_layers is not None
                    or self.config.last_pipeline_num_layers is not None
                ):
                    # Calculate number of pipelines for distributing layers
                    middle_pipeline_stages = parallel_state.get_pipeline_model_parallel_world_size()
                    middle_pipeline_stages -= sum(
                        [
                            1 if x is not None else 0
                            for x in (
                                self.config.first_pipeline_num_layers,
                                self.config.last_pipeline_num_layers,
                            )
                        ]
                    )

                    # Calculate layers to distribute
                    first_pipeline_offset = (
                        0
                        if self.config.first_pipeline_num_layers is None
                        else self.config.first_pipeline_num_layers
                    )
                    last_pipeline_offset = (
                        0
                        if self.config.first_pipeline_num_layers is None
                        else self.config.last_pipeline_num_layers
                    )

                    middle_num_layers = (
                        self.config.num_layers - first_pipeline_offset - last_pipeline_offset
                    )

                    if middle_pipeline_stages > 0:
                        num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                    else:
                        num_layers_per_pipeline_rank = 0

                    middle_pipeline_rank = (
                        pipeline_rank
                        if self.config.first_pipeline_num_layers is None
                        else pipeline_rank - 1
                    )

                    if pipeline_rank == 0:
                        offset = 0
                    else:
                        offset = (
                            middle_pipeline_rank * num_layers_per_pipeline_rank
                        ) + first_pipeline_offset
                else:
                    offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0

        return offset

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_states,
        value_states,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        """
        Perform a forward pass through the dragon layer.

        This method implements the core computation of a dragon layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            inference_params (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        # Residual connection.
        residual = hidden_states
        
        
        # Input layernorm.
        input_layernorm_output = self.input_layernorm(hidden_states)
        

        # Input projection.
        """
        input_proj_output = self.input_proj(hidden_states)
            
        hidden_states, gate, query_states, kv_states = input_proj_output.tensor_split(
            (
                self.intermediate_size,  # First split for hidden_states
                self.intermediate_size * 2,  # Split for gate
                self.intermediate_size * 3,  # Split for query_states
            ),
            dim=1
        )
        if self.cache_sharing == CacheSharing.FIRST:
            key_states, value_states = kv_states.tensor_split(
                (
                    self.attention_head_size * self.num_query_groups,  # Split for key_states
                ),
                dim=1
            )
        """
        # Self attention.
        attention_output = self.self_attention(
            input_layernorm_output,
            #query_states, #When used input_proj we will directly use query_states
            key_states,
            value_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        # print("Attention Output: ", attention_output.shape)
        # Post attention layernorm.
        # print("Attention Output contig: ", attention_output.is_contiguous())
        attention_layer_norm = self.self_attn_layernorm(attention_output)

        mamba_output = self.mamba_mixer(
            input_layernorm_output,
            attention_mask=attention_mask, #not used but required for compatibility
            inference_params=inference_params,
        )
        # print("Mamba Output: ", mamba_output.shape)
        # Post Mamba layernorm.
        mamba_layer_norm = self.mamba_layernorm(mamba_output)
        
        # Output projection.
        average  = (attention_layer_norm + mamba_layer_norm) / 2
        
        output_proj_output, _ = self.output_projection(average)
        # print("Output Projection: ", output_proj_output)
        
        residual = output_proj_output + residual

        # print("Residual after output projection: ", residual.shape)
        
        
        # MLP.
        mlp_output, _ = self.mlp(residual)
        
        output = mlp_output + residual
        
        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        if self.cache_sharing == CacheSharing.SECOND:
            return output, None, None
        return output, key_states, value_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the transformer layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'cudagraph_manager'):
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        return self.mamba_mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)