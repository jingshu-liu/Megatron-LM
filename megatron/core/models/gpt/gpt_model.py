# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict
from typing import Dict, Literal, Optional

from torch import Tensor
import torch
import torch.nn.functional as F
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.tensor_parallel.mappings import scatter_to_tensor_model_parallel_region
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from cut_cross_entropy import linear_cross_entropy
import torch.distributed as dist

class GPTModel(LanguageModule):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        self.patch_size = self.config.patch_size
        self.use_cce = self.config.use_cce
        self.tensor_model_parallel_size = self.config.tensor_model_parallel_size
        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None
            gather_output = not self.parallel_output
            if self.use_cce :
                skip_weight_param_allocation = False
            else:
                skip_weight_param_allocation = self.pre_process
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=gather_output,
                skip_weight_param_allocation=skip_weight_param_allocation
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )
            # self.output_layer = torch.nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        batch_size, length = input_shape
        device = inputs_embeds.device
        dtype=inputs_embeds.dtype

        mask = torch.triu(torch.ones(length, length, device=device), diagonal=1).to(dtype)
        mask = mask.masked_fill(mask == 1, torch.finfo(dtype).min)
        mask = mask[None, None, :, :].expand(batch_size, 1, length, length)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(length, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        return mask

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None
        # print('---------input------', decoder_input.size(), input_ids.size())
        # apply patch level here, todo deactivate Megatron GPTdataset preshifting
        if self.patch_size>1:
            # use decoder_input to infer the size because with tensor parallel the input tensor is splitted while attention mask and input ids are not 
            # won't be different if sequence_parallel = False    
            batch_size, seq_length = decoder_input.size()[1], decoder_input.size()[0]
            num_patches = seq_length // self.patch_size

            # s b h
            decoder_input = decoder_input.view(num_patches, self.patch_size,batch_size, -1).mean(1)
            
            global_seq_length = input_ids.size()[1]
            global_num_patches = global_seq_length // self.patch_size
            position_ids = position_ids[:, :global_num_patches]
            if attention_mask is not None:
                # we don't have past_key_values here it's for generation, but we have it in llama_modeling, currently we set it to 0 todo: verify
                # make sure we don't use patch level for inference .
                past_key_values_length = 0
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, num_patches), decoder_input, past_key_values_length
                )
        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        if self.patch_size>1 and self.use_cce:
            if output_weight is None:
                output_weight = self.output_layer.weight
            if self.tensor_model_parallel_size>1:
                if  not self.share_embeddings_and_output_weights:
                    full_weight = gather_full_weight(self.output_layer, self.tensor_model_parallel_size)
                else:
                    full_weight = tensor_parallel.gather_from_tensor_model_parallel_region(output_weight.transpose(0,1)) # gather along the last dimension so we need to transpose the matrix to gather along the vocab dim
                    full_weight = full_weight.transpose(0,1)
                output_weight = full_weight
                
            # if labels.device == torch.device("cuda:0"):
            #     print('hidden states ', hidden_states[-32:,0,:10])
            if input_ids.size()[1]//self.patch_size > hidden_states.size()[0]:
                hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states)
            # hidden_states = tensor_parallel.gather_from_tensor_model_parallel_region(hidden_states)
            
            hidden_states = hidden_states[:-1, ...] # s b h
            hidden_states = hidden_states.transpose(0,1)
            # local_labels = scatter_to_tensor_model_parallel_region(labels) #  along the last dim
            
            shift_labels = labels[..., self.patch_size:].view(batch_size, -1, self.patch_size) # b s patch 
            patch_labels = shift_labels
            # print("e, c, target size in cce: ", hidden_states.size(), output_weight.size(), patch_labels[:,:, 0].size())
            
            patch_loss_list = []
            for i in range(self.patch_size):
                patch_loss = linear_cross_entropy(hidden_states, output_weight, patch_labels[:,:, i], shift=False, impl="cce", reduction='none')
                patch_loss_list.append(patch_loss)
            return patch_loss_list
        # check_tensor_parallel_groups(self.tensor_model_parallel_size)

        if self.use_cce:       
            if output_weight is None:
                output_weight = self.output_layer.weight # vocab_size, hidden_size
            if self.tensor_model_parallel_size>1:
                if  not self.share_embeddings_and_output_weights:
                    full_weight = gather_full_weight(self.output_layer, self.tensor_model_parallel_size)
                else:
                    full_weight = tensor_parallel.gather_from_tensor_model_parallel_region(output_weight.transpose(0,1)) # gather along the last dimension so we need to transpose the matrix to gather along the vocab dim
                    full_weight = full_weight.transpose(0,1) # back to shape (vocab, dim)
                output_weight = full_weight
            # since cce is only used during the training, we don't need to consider the inference
            # if labels is None: 
                # labels = input_ids
            if input_ids.size()[1] > hidden_states.size()[0]:
                hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states) # gather from the 1st dimension, in this case, sequence length
            # no need to do shifting because Megatron GPTdataset already does it 
            # tokens = text[:-1].contiguous()
            # labels = text[1:].contiguous()
            loss = linear_cross_entropy(hidden_states.transpose(0,1), output_weight, labels, shift=False, impl="cce", reduction='none')
            return loss
        logits, _ = self.output_layer(hidden_states, weight=output_weight)
        
        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1)

        if self.patch_size>1:        
            
            # with Megatron self.compute_language_model_loss
            batch_size, global_seq_length = input_ids.size()
            global_num_patches = global_seq_length // self.patch_size
            shift_logits = logits[:-1, ...] # s b h
            shift_labels = labels[..., self.patch_size:].view(batch_size, global_num_patches-1, self.patch_size) # b s patch 
            patch_loss_list = []
            for i in range(self.patch_size):
                patch_loss = self.compute_language_model_loss(shift_labels[:,:,i], shift_logits) 
                patch_loss_list.append(patch_loss)
            
            return patch_loss_list # [b s]
        
        loss = self.compute_language_model_loss(labels, logits) # [s b] => [b, s]
        
        #### pytorch cross entropy
        # ce = torch.nn.CrossEntropyLoss(reduction='none')
        # # Reshape logits to (batch_size * sequence_length, num_classes)
        # logits = logits.transpose(0,1).contiguous()
        # logits = logits.view(-1, self.vocab_size)

        # # Reshape labels to (batch_size * sequence_length)
        # labels = labels.view(-1)
        # loss = ce(logits, labels)
        # loss = loss.view(input_ids.size()[0], self.max_sequence_length) # loss per token
        
        return loss

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility
        (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the
        # _extra_state key but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        return sharded_state_dict


def gather_full_weight(layer, tensor_model_parallel_size):
    if layer.weight is None:
        raise ValueError("layer.weight is None. Ensure skip_weight_param_allocation=False or pass weight explicitly.")

    rank = dist.get_rank()  # Current device rank

    world_size = dist.get_world_size()  # Total GPUs
    # tp_group_size = world_size // tensor_model_parallel_size 
    tp_group_size = tensor_model_parallel_size # how many gpus for one tp group
    # Ensure we're in the correct setting
    if world_size % tp_group_size != 0:
        raise ValueError(f"Invalid tensor_model_parallel_size={tp_group_size} for world_size={world_size}")

    # Get tensor-parallel group index
    tp_group_id = rank // tp_group_size  # e.g., {0,1} -> group 0, {2,3} -> group 1
    tp_group_ranks = list(range(tp_group_id * tp_group_size, (tp_group_id + 1) * tp_group_size)) # group id 0 will have a list [0,1]

    # Create tensor-parallel process group
    tp_group = dist.new_group(tp_group_ranks)

    # Local weight chunk (each GPU holds part of the full weight)
    local_weight = layer.weight  

    # Gather weight tensors across the tensor-parallel group
    gathered_weights = [torch.empty_like(local_weight) for _ in range(tp_group_size)]
    
    dist.all_gather(gathered_weights, local_weight, group=tp_group)
    gathered_weights = torch.cat(gathered_weights, dim=0)
    return gathered_weights

def check_tensor_parallel_groups(tensor_parallel_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get tensor parallel size from your config
    num_tensor_parallel_groups = world_size // tensor_parallel_size

    # Create tensor parallel groups manually
    tensor_parallel_groups = []
    for i in range(num_tensor_parallel_groups):
        ranks = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))
        group = dist.new_group(ranks)
        tensor_parallel_groups.append(group)

        if rank in ranks:
            print(f"Rank {rank} is in tensor parallel group: {ranks}")

def print_loss(loss, rank=0):
    if dist.is_initialized() and dist.get_rank() == rank:
        print(loss)