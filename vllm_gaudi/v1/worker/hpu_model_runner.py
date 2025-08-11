# SPDX-License-Identifier: Apache-2.0
import collections
import contextlib
import functools
import itertools
import math
import os
import gc
import time
from dataclasses import dataclass, field, fields
from enum import IntEnum
from array import array
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeAlias, TypeVar, Union, NamedTuple
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling, is_pooling_model, is_text_generation_model)
from vllm.distributed.parallel_state import get_world_group

import habana_frameworks.torch as htorch
import habana_frameworks.torch.internal.bridge_config as bc
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import vllm_gaudi.extension.environment as environment
import vllm_gaudi.extension.bucketing.linear as linear
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm_gaudi.extension.bucketing.common import HPUBucketingManager
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm_gaudi.extension.profiler import (HabanaHighLevelProfiler,
                                           HabanaMemoryProfiler,
                                           HabanaProfilerCounterHelper,
                                           format_bytes)
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceData, SequenceGroupMetadata,
                           SequenceOutput)
from vllm_gaudi.extension.runtime import get_config

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.attention.selector import get_attn_backend
from vllm.config import (VllmConfig, update_config, DeviceConfig)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader import get_model, get_model_loader
from vllm.sampling_params import SamplingType
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, cdiv,
                        is_pin_memory_available, make_tensor_with_pad)
from vllm_gaudi.utils import is_fake_hpu
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.utils import bind_kv_cache
from vllm_gaudi.v1.worker.hpu_input_batch import InputBatch
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.distributed.parallel_state import get_pp_group

from vllm.model_executor.models.interfaces import supports_transcription
from vllm.model_executor.models.interfaces_base import (
    is_pooling_model, is_text_generation_model)
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

_TYPE_CACHE: dict[str, dict[str, Any]] = {}

_PAD_SLOT_ID = 0
_PAD_BLOCK_ID = 0

class BucketingFailedException(Exception):
    pass


def setup_profiler(warmup, active):
    schedule = torch.profiler.schedule(wait=0,
                                       warmup=warmup,
                                       active=active,
                                       repeat=1)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.HPU
    ]
    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.',
                                                                use_gzip=True),
        record_shapes=False,
        with_stack=True)
    return profiler


@dataclass
class PromptDecodeInfo:
    prompt_req_ids: list[str]
    decode_req_ids: list[str]
    prompt_scheduled_tokens: list[int]


@dataclass
class PromptData:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: HPUAttentionMetadataV1


@dataclass
class DecodeData:
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional[HPUAttentionMetadataV1] = None


empty_list: Callable[[], list] = lambda: field(default_factory=list)


@dataclass
class BatchContents:
    req_ids: list[str] = empty_list()
    token_ids: list[list[int]] = empty_list()
    context_lens: list[int] = empty_list()
    blocks: list[list[int]] = empty_list()
    logits_positions: list[list[int]] = empty_list()

    def get_num_tokens(self):
        return [len(t) for t in self.token_ids]


#TODO(kzawora): remove this
@dataclass
class PrefillInputData:
    request_ids: list = empty_list()
    prompt_lens: list = empty_list()
    token_ids: list = empty_list()
    position_ids: list = empty_list()
    attn_metadata: list = empty_list()
    logits_indices: list = empty_list()
    logits_requests: list = empty_list()


#TODO(kzawora): remove this
@dataclass
class DecodeInputData:
    num_decodes: int
    token_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    attn_metadata: Optional[HPUAttentionMetadataV1] = None
    logits_indices: Optional[torch.Tensor] = None

TModelInputForHPU = TypeVar('TModelInputForHPU', bound="ModelInputForHPU")
  
@dataclass(frozen=True)
class ModelInputForHPU(ModelRunnerInputBase):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    # lora_mapping: Optional["LoRAMapping"] = None
    # lora_requests: Optional[Set[LoRARequest]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[Dict[str, torch.Tensor]] = None
    real_batch_size: Optional[int] = None
    batch_size_padded: Optional[int] = None
    virtual_engine: int = 0
    lora_ids: Optional[List[int]] = None
    async_callback: Optional[Callable] = None
    is_first_multi_step: bool = True
    is_last_step: bool = True

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "real_batch_size": self.real_batch_size,
            "batch_size_padded": self.batch_size_padded,
            "virtual_engine": self.virtual_engine,
            "lora_ids": self.lora_ids,
            "is_first_multi_step": self.is_first_multi_step,
            "is_last_step": self.is_last_step,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: type[TModelInputForHPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForHPU:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


def bool_helper(value):
    value = value.lower()
    return value in ("y", "yes", "t", "true", "on", "1")


Mergeable: TypeAlias = Union[BatchContents, PrefillInputData]


def shallow_tuple(obj: Mergeable) -> tuple:
    """Returns a shallow tuple with dataclass field values"""
    # Unfortunately dataclasses.astuple deepcopies the data
    # se we can't use it
    return tuple(getattr(obj, field.name) for field in fields(obj))


def merge_contents(lhs: Mergeable, *rhs: Mergeable):
    """Extends all internal lists of a dataclass with """
    """values from given objects"""
    lhs_type = type(lhs)
    lhs_tuple = shallow_tuple(lhs)
    for other in rhs:
        assert lhs_type is type(other),\
            'Only objects of the same type can be merged'
        for dst, src in zip(lhs_tuple, shallow_tuple(other)):
            dst.extend(src)


def flatten(in_list):
    """Return a flattened representation of a list"""
    return list(itertools.chain(*in_list))


def gather_list(input, indices, v):
    """Gather values from input using indices"""
    return [input[i] if i is not None else v for i in indices]


def _async_h2d_tensor(data, dtype, device='hpu'):
    return torch.tensor(data, dtype=dtype, device='cpu').to(device,
                                                            non_blocking=True)


def _async_h2d_tensor_copy(source, device='hpu'):
    assert source.device.type == 'cpu', \
        "Source tensor is not present in host memory!"
    target = torch.empty(source.shape, dtype=source.dtype, device=device)
    target.copy_(source, non_blocking=True)
    return target


def ensure_decodes_first(b: InputBatch):
    num_reqs = b.num_reqs
    while True:
        # Find the first prompt index
        first_prompt_index = None
        for i in range(num_reqs):
            if b.num_computed_tokens_cpu[i] < b.num_prompt_tokens[i]:
                first_prompt_index = i
                break
        if first_prompt_index is None:
            break

        # Find the last decode index
        last_decode_index = None
        for i in reversed(range(num_reqs)):
            if b.num_computed_tokens_cpu[i] >= b.num_prompt_tokens[i]:
                last_decode_index = i
                break
        if last_decode_index is None:
            break

        # Sanity
        assert first_prompt_index != last_decode_index

        # Check if done
        if first_prompt_index > last_decode_index:
            break

        # Swap
        b.swap_states(first_prompt_index, last_decode_index)


def get_target_layer_suffix_list(model_type) -> list[str]:
    # This sets the suffix for the hidden layer name, which is controlled by
    # VLLM_CONFIG_HIDDEN_LAYERS. The default suffix is "DecoderLayer," which is
    # applicable for most language models such as LLaMA, Qwen, and BART. If the
    # model's decoder layer name differs from the default, it will need to
    # be specified here.
    decoder_layer_table = {
        "gpt_bigcode": "BigCodeBlock",
    }

    return [
        decoder_layer_table.get(model_type, "DecoderLayer"), "EncoderLayer"
    ]


def modify_model_layers(module: torch.nn.Module,
                        suffix_list: list[str],
                        n=1,
                        counter=None):
    """Currently add mark_step at the end of specified layers.
    """

    def forward_hook(module, args, output):
        htorch.core.mark_step()
        return output

    if counter is None:
        counter = [0]

    for child_name, child_module in module.named_children():
        if any(
                child_module.__class__.__name__.endswith(layer)
                for layer in suffix_list):
            counter[0] += 1
            if counter[0] % n == 0:
                child_module.register_forward_hook(forward_hook)
        else:
            modify_model_layers(child_module, suffix_list, n, counter)


class HpuModelAdapter(torch.nn.Module):

    def __init__(self, model, vllm_config):
        super().__init__()
        self.model = model
        self.prefill_use_fusedsdpa = get_config(
        ).prompt_attn_impl == 'fsdpa_impl'
        self.recompute_cos_sin = os.getenv('VLLM_COS_SIN_RECOMPUTE',
                                           'false').lower() in ['1', 'true']
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.dtype = vllm_config.model_config.dtype
        self._rotary_embed_module = self._get_rotary_embedding_module(
            self.model)
        self._rotary_prepare_cos_sin = self._get_prepare_cos_sin()

    def _get_rotary_embedding_module(self, model: torch.nn.Module):
        """
        Dynamically get the RotaryEmbedding layer in the model.
        This function will recursively search through the module 
        hierarchy to find and return a RotaryEmbedding layer.
        If no such layer is found, it returns None.
        """
        if model is None:
            return None

        if model.__class__.__name__.endswith("RotaryEmbedding"):
            return model

        if hasattr(model, 'children'):
            for child in model.children():
                result = self._get_rotary_embedding_module(child)
                if result is not None:
                    return result
        return None

    def _get_prepare_cos_sin(self):
        if self._rotary_embed_module is not None and hasattr(
                self._rotary_embed_module, 'prepare_cos_sin'):
            return self._rotary_embed_module.prepare_cos_sin
        return None

    def _reset_rotary_cos_sin(self):
        if hasattr(self._rotary_embed_module, "cos"):
            delattr(self._rotary_embed_module, "cos")
        if hasattr(self._rotary_embed_module, "sin"):
            delattr(self._rotary_embed_module, "sin")

    def _set_attn_bias(self, attn_metadata, batch_size, seq_len, device,
                       dtype):
        if (attn_metadata is None or
            (self.prefill_use_fusedsdpa and attn_metadata.block_list is None)
                or not attn_metadata.is_prompt):
            return attn_metadata

        if attn_metadata.attn_bias is not None:
            return attn_metadata

        prefill_metadata = attn_metadata

        seq_lens_t = prefill_metadata.seq_lens_tensor
        context_lens_t = prefill_metadata.context_lens_tensor

        block_list = attn_metadata.block_list
        max_context_len = (block_list.size(-1) //
                           batch_size if block_list is not None else 0)
        max_context_len = max_context_len * self.block_size
        past_mask = torch.arange(0,
                                 max_context_len,
                                 dtype=torch.int32,
                                 device=device)
        past_mask = (past_mask.view(1, -1).expand(batch_size, -1).ge(
            context_lens_t.view(-1, 1)).view(batch_size, 1, -1).expand(
                batch_size, seq_len, -1).view(batch_size, 1, seq_len, -1))

        len_mask = (torch.arange(0, seq_len, device=device,
                                 dtype=torch.int32).view(1, seq_len).ge(
                                     seq_lens_t.unsqueeze(-1)).view(
                                         batch_size, 1, 1, seq_len))
        causal_mask = torch.triu(torch.ones((batch_size, 1, seq_len, seq_len),
                                            device=device,
                                            dtype=torch.bool),
                                 diagonal=1)
        mask = causal_mask.logical_or(len_mask)
        mask = torch.concat((past_mask, mask), dim=-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))
        attn_metadata = custom_tuple_replace(prefill_metadata,
                                             "TrimmedAttentionMetadata",
                                             attn_bias=attn_bias)
        return attn_metadata

    def _set_block_mapping(self, metadata, batch_size, device, dtype):
        mask = torch.arange(0,
                            self.block_size,
                            device=device,
                            dtype=torch.int32).unsqueeze(0)
        mask = mask >= metadata.block_usage.unsqueeze(-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))

        if not is_fake_hpu():
            block_mapping = torch.nn.functional.one_hot(metadata.block_groups,
                                                        num_classes=batch_size)
        else:
            # Unfortunately one_hot on CPU
            # doesn't handle out of bounds classes so we need to convert
            # all negative values to 0 (block_mapping) or bs (block_groups)
            block_groups = metadata.block_groups.to(torch.long)
            block_mapping = torch.nn.functional.relu(block_groups)
            block_mapping = torch.nn.functional.one_hot(block_mapping,
                                                        num_classes=batch_size)
            oob_values = block_groups.lt(0)
            block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
            block_groups.masked_fill_(oob_values, batch_size)
            metadata = custom_tuple_replace(metadata,
                                            "TrimmedAttentionMetadata",
                                            block_groups=block_groups)
        block_mapping = block_mapping.to(dtype)
        metadata = custom_tuple_replace(metadata,
                                        "TrimmedAttentionMetadata",
                                        block_mapping=block_mapping,
                                        attn_bias=attn_bias)
        return metadata

    def _update_metadata(self, attn_metadata, batch_size, seq_len, device,
                         dtype):
        if attn_metadata.is_prompt:
            attn_metadata = self._set_attn_bias(attn_metadata, batch_size,
                                                seq_len, device, dtype)
        else:
            attn_metadata = self._set_block_mapping(attn_metadata, batch_size,
                                                    device, dtype)
        return attn_metadata

    def forward(self, *args, **kwargs):
        # TODO(kzawora): something goes VERY WRONG when operating on
        # kwargs['attn_metadata'].slot_mapping, compared to untrimmed metadata
        kwargs = kwargs.copy()
        #        selected_token_indices = kwargs.pop('selected_token_indices')
        if 'warmup_mode' in kwargs:
            kwargs.pop('warmup_mode')
        input_ids = kwargs['input_ids']
        kwargs['attn_metadata'] = self._update_metadata(
            kwargs['attn_metadata'], input_ids.size(0), input_ids.size(1),
            input_ids.device, self.dtype)
        if self._rotary_prepare_cos_sin is not None:
            self._rotary_prepare_cos_sin(
                kwargs['positions'], recompute_cos_sin=self.recompute_cos_sin)
        attn_meta = kwargs.pop('attn_metadata')
        if 'kv_caches' in kwargs:
            kwargs.pop('kv_caches')
        with set_forward_context(attn_meta, self.vllm_config):
            hidden_states = self.model(*args, **kwargs)
            if self._rotary_prepare_cos_sin is not None:
                self._reset_rotary_cos_sin()
        return hidden_states

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    # def sample(self, *args, **kwargs):
    #    return self.sampler(*args, **kwargs)

    def generate_proposals(self, *args, **kwargs):
        return self.model.generate_proposals(*args, **kwargs)

    # sampler property will be used by spec_decode_worker
    # don't rename
    # @property
    # def sampler(self):
    #    return self.model.sampler


def _maybe_wrap_in_hpu_graph(*args, **kwargs):
    return htorch.hpu.wrap_in_hpu_graph(
        HpuModelAdapter(*args, **kwargs), disable_tensor_cache=True
    ) if htorch.utils.internal.is_lazy() else HpuModelAdapter(*args, **kwargs)


def subtuple(obj: object,
             typename: str,
             to_copy: list[str],
             to_override: Optional[dict[str, object]] = None):
    if obj is None:
        return None
    if to_override is None:
        to_override = {}
    fields = set(to_copy) | set(to_override.keys())
    if type(obj) is dict:
        values = {key: obj[key] for key in fields if key in obj}
    else:
        values = {f: to_override.get(f, getattr(obj, f)) for f in fields}
    if typename not in _TYPE_CACHE:
        _TYPE_CACHE[typename] = {
            'type': collections.namedtuple(typename, ' '.join(fields)),
            'fields': fields
        }
    return _TYPE_CACHE[typename]['type'](**values)  # type: ignore


def custom_tuple_replace(obj: object, typename: str, **to_override):
    # Torch compile dynamo doesn't support calling any named tuple
    # dynamic methods other than len and get_attr. This function is
    # a torch.compile friendly version of tuple._replace

    cached_type = _TYPE_CACHE[typename]['type']
    fields = _TYPE_CACHE[typename]['fields']
    values = {
        field: getattr(obj, field)
        for field in fields  # type: ignore
    }
    values.update(to_override)
    return cached_type(**values)  # type: ignore


def trim_attn_metadata(metadata: HPUAttentionMetadataV1) -> object:
    # NOTE(kzawora): To anyone working on this in the future:
    # Trimming metadata is required when using HPUGraphs.
    # Attention metadata is going to be hashed by PT bridge, and
    # appropriate HPUGraphs will be matched based on all inputs' hash.

    # Before you put more keys in here, make sure you know their
    # value type and make sure you know how it's going to be hashed.
    # You can find that information in input_hash function
    # in habana_frameworks/torch/hpu/graphs.py. You can also hash
    # it manually with torch.hpu.graphs.input_hash(attention_metadata)

    # If you use primitive types here - they will get hashed based
    # on their value. You *will* get lots of excessive graph captures
    # (and an OOM eventually) if you decide to put something like
    # seq_len int here.
    # If you absolutely need a scalar, put it in a tensor. Tensors
    # get hashed using their metadata, not their values:
    # input_hash(torch.tensor(123)) == input_hash(torch.tensor(321))
    # input_hash(123) != input_hash(321)
    # input_hash("abc") != input_hash("cba")
    attention_metadata = subtuple(metadata, 'TrimmedAttentionMetadata', [
        'attn_bias', 'seq_lens_tensor', 'context_lens_tensor', 'block_list',
        'block_mapping', 'block_usage', 'slot_mapping', 'is_prompt',
        'block_size', 'block_groups'
    ])
    return attention_metadata


def next_pow2(value: int, base: int):
    res = base
    while value > 1:
        value = (value + 1) // 2
        res *= 2
    return res


def round_up(value: int, k: int):
    return (value + k - 1) // k * k


def pad_list(input, target_len, val_generator):
    padding = target_len - len(input)
    if padding > 0:
        input.extend(itertools.islice(val_generator, padding))
    return input

# How batches are constructed.
class BatchType(IntEnum):
    # Every batch is prefill.
    PREFILL = 0
    # Every batch is decode.
    DECODE = 1
    # Batch is a mixture of prefill and decode.
    MIXED = 2


TModelInputForHPU = TypeVar('TModelInputForHPU', bound="ModelInputForHPU")

class HPUModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device = 'hpu',
        is_driver_worker: bool = False,
    ):
        # TODO: use ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        environment.set_vllm_config(vllm_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.is_driver_worker = is_driver_worker

        self.sampler = get_sampler()

        # NOTE(kzawora) update_env is a hack to work around VLLMKVCache in
        # hpu-extension which selects fetch_from_cache implementation based
        # on env vars... this should be fixed in the future
        self.enable_bucketing = get_config().use_bucketing
        self.use_contiguous_pa = get_config().use_contiguous_pa
        self.skip_warmup = get_config().skip_warmup

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Model-related.
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, LayerBlockType.attention)
        self.num_query_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.num_kv_heads = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.hidden_size = self.model_config.get_hidden_size()

        self.attn_backend = get_attn_backend(
            self.head_size,
            self.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        )

        # Lazy initialization
        # self.model: nn.Module  # set after load_model
        self.kv_caches: list[torch.Tensor] = []
        self.inc_initialized_successfully = False
        self._is_inc_finalized = False

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size])
        self.mem_margin = None

        self.use_hpu_graph = not self.model_config.enforce_eager
        self.max_batch_size = self.scheduler_config.max_num_seqs
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.max_prefill_batch_size = 1  # TODO(kzawora): add knob for that
        self.seen_configs: set = set()
        self.max_num_batched_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.use_merged_prefill = get_config().merged_prefill
        self.use_prefix_caching = (
            self.vllm_config.cache_config.enable_prefix_caching)
        self.bucketing_manager = HPUBucketingManager()
        if self.enable_bucketing:
            logger.info("Bucketing is ON.")
            self.bucketing_manager.initialize(
                max_num_seqs=self.max_num_seqs,
                max_num_prefill_seqs=self.max_prefill_batch_size,
                block_size=self.block_size,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_model_len=self.max_model_len)
            self.graphed_buckets: set[Any] = set()
        else:
            logger.info("Bucketing is OFF.")
        self._PAD_SLOT_ID = -1
        self._PAD_BLOCK_ID = -1
        self._tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            lora_config=vllm_config.lora_config).tokenizer

        # TODO(madamczyk-intel): add a knob for that
        # TODO(madamczyk-intel): debug why increasing it lowers acc
        self.logits_rounding = 1
        # High-level profiler
        self.profiler = HabanaHighLevelProfiler()
        self.profiler_counter_helper = HabanaProfilerCounterHelper()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in forward_ctx.items():
            if isinstance(attn_module, FusedMoE):
                continue

            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            pooling_params = new_req_data.pooling_params
            if new_req_data.sampling_params is not None:
                sampling_params = new_req_data.sampling_params
                if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if pooling_params:
                assert (task := pooling_params.task) is not None, (
                    "You did not set `task` in the API")

                model = cast(VllmModelForPooling, self.model)
                # to_update = model.pooler.get_pooling_updates(task)
                # to_update.apply(pooling_params)
            sampling_params=None
            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)
        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if not resumed_from_preemption:
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
            else:
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                # Add spec_token_ids to token_ids_cpu.
                spec_token_ids = \
                    scheduler_output.scheduled_spec_decode_tokens.get(
                        req_id, ())
                if spec_token_ids:
                    start_index = end_token_index
                    end_token_index += len(spec_token_ids)
                    self.input_batch.token_ids_cpu[
                        req_index,
                        start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec decode tokens
                self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()
        return batch_changed

    def get_model(self) -> torch.nn.Module:
        assert self.model is not None
        return self.model

    def _get_prompts_and_decodes(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> PromptDecodeInfo:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Traverse decodes first
        decode_req_ids = []
        num_computed_tokens_decode = []
        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            if num_computed_tokens < num_prompt_tokens:
                # This is prompt
                break

            # This is decode
            assert num_scheduled_tokens == 1
            decode_req_ids.append(req_id)
            num_computed_tokens_decode.append(int(num_computed_tokens + 1))

        if self.profiler.enabled:
            self.profiler_counter_helper.capture_decode_seq_stats(
                num_computed_tokens_decode)

        # Traverse prompts
        prompt_req_ids = []
        prompt_scheduled_tokens = []
        for i in range(len(decode_req_ids), num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            # Must be prompt
            assert num_computed_tokens < num_prompt_tokens
            num_output_tokens = len(self.requests[req_id].output_token_ids)
            assert num_output_tokens == 0, \
                f'req_id: {req_id}, {num_output_tokens}'

            prompt_req_ids.append(req_id)
            prompt_scheduled_tokens.append(num_scheduled_tokens)

        return PromptDecodeInfo(prompt_req_ids, decode_req_ids,
                                prompt_scheduled_tokens)

    def _prepare_sampling(self,
                          batch_changed: bool,
                          request_ids: Union[None, list[str]] = None,
                          pad_to: Optional[int] = None) -> SamplingMetadata:
        # Create the sampling metadata.
        req_id_output_token_ids: dict[str, list[int]] = \
            {req_id: req.output_token_ids \
                for req_id, req in self.requests.items()}
        if request_ids is not None:
            req_id_output_token_ids = {
                req_id: req_id_output_token_ids[req_id] \
                    for req_id in request_ids}
        req_id_output_token_ids_lst = list(req_id_output_token_ids.items())
        if pad_to is not None:
            while len(req_id_output_token_ids_lst) < pad_to:
                req_id_output_token_ids_lst.append(
                    req_id_output_token_ids_lst[0])
        sampling_metadata = self.input_batch.make_selective_sampling_metadata(
            req_id_output_token_ids_lst, skip_copy=not batch_changed)
        return sampling_metadata

    def get_habana_paged_attn_buffers(self, block_tables, slot_mapping,
                                      batch_size):
        last_block_usage = [
            slot[0] % self.block_size + 1 for slot in slot_mapping
        ]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
        block_usage = [[self.block_size] * (len(bt) - 1) + [lbu]
                       for bt, lbu in zip(block_tables, last_block_usage)
                       if bt]
        block_list = flatten(block_tables)
        block_groups = flatten(block_groups)
        block_usage = flatten(block_usage)
        assert len(block_list) == len(block_groups)
        assert len(block_list) == len(block_usage)

        padding_fn = None
        block_bucket_size: int
        if self.use_contiguous_pa:
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            block_bucket_size = \
                self.bucketing_manager.find_decode_bucket(batch_size,
                                                          block_bucket_size)[2]
            indices: list[Any]
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(
                tensor, indices, pad_value)
        else:
            block_bucket_size = \
                self.bucketing_manager.find_decode_bucket(batch_size,
                                                          len(block_list))[2]
            padding_fn = lambda tensor, pad_value: pad_list(
                tensor, block_bucket_size, itertools.repeat(pad_value))

        block_list = padding_fn(block_list, self._PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        block_list = torch.tensor(block_list, dtype=torch.long, device='cpu')
        block_groups = torch.tensor(block_groups,
                                    dtype=torch.long,
                                    device='cpu')
        block_usage = torch.tensor(block_usage,
                                   dtype=self.model_config.dtype,
                                   device='cpu')
        return block_list, block_groups, block_usage

    def _align_and_pad(self, data, bucketing, padding_gen):
        bs = len(data)
        target_bs, target_len = bucketing
        if target_bs == 1 and bs > 1:
            data = [list(itertools.chain(*data))]
        data = [pad_list(x, target_len, padding_gen) for x in data]
        padding = itertools.islice(padding_gen, target_len)
        data = pad_list(data, target_bs, itertools.repeat(padding))
        return data

    def _bucketize_merged_prompt(self, seq_lens, num_blocks):
        seq = sum(seq_lens)
        num_blocks = sum(num_blocks)
        seq = self.bucketing_manager.find_prompt_bucket(1, seq, num_blocks)[1]
        num_blocks = round_up(num_blocks, 32)
        return (1, seq, num_blocks)

    def _bucketize_2d_prompt(self, seq_lens, num_blocks):
        bs = len(seq_lens)
        if bs > self.max_prefill_batch_size:
            raise BucketingFailedException
        seq = max(seq_lens)
        num_blocks = max(num_blocks) if len(num_blocks) > 0 else 0
        bs, seq, num_blocks = self.bucketing_manager.find_prompt_bucket(
            bs, seq, num_blocks)
        return (bs, seq, num_blocks)

    def _get_prompt_bucketing_fn(self):
        if self.use_merged_prefill:
            return self._bucketize_merged_prompt
        else:
            return self._bucketize_2d_prompt

    def _can_merge_prefill_contents(self, lhs, rhs):
        combined_num_tokens = lhs.get_num_tokens() + rhs.get_num_tokens()
        bucketing_fn = self._get_prompt_bucketing_fn()
        try:
            target_bs, target_seq, target_blocks = bucketing_fn(
                combined_num_tokens, [])
        except BucketingFailedException:
            return False
        target_bs, target_seq, target_blocks = bucketing_fn(
            combined_num_tokens, [])
        return target_bs <= self.max_prefill_batch_size and\
            target_bs * target_seq <= self.max_num_tokens

    def _extract_prefill_batch_contents(self, num_prefills, num_decodes,
                                        num_scheduled_tokens):
        # DECODES are the first num_decodes REQUESTS.
        # PREFILLS are the next num_reqs - num_decodes REQUESTS.
        num_reqs = num_prefills + num_decodes
        block_table_cpu_tensor = self.input_batch.block_table[
            0].get_cpu_tensor()
        all_batch_contents = [BatchContents()]

        for batch_idx in range(num_decodes, num_reqs):
            req_id = self.input_batch.req_ids[batch_idx]
            context_len = self.input_batch.num_computed_tokens_cpu[batch_idx]
            query_len = num_scheduled_tokens[batch_idx]

            token_ids = self.input_batch.token_ids_cpu[
                batch_idx, context_len:context_len + query_len].tolist()

            num_blocks = round_up(context_len + query_len,
                                  self.block_size) // self.block_size
            blocks = block_table_cpu_tensor[batch_idx, :num_blocks].tolist()

            prompt_tokens = self.input_batch.num_prompt_tokens[batch_idx]
            #TODO: Fix non-prompt case
            num_output_logits = context_len + query_len - prompt_tokens + 1
            logits_positions = list(
                range(query_len - num_output_logits, query_len))

            new_batch_contents = BatchContents(
                req_ids=[req_id],
                token_ids=[token_ids],
                context_lens=[context_len],
                blocks=[blocks],
                logits_positions=[logits_positions],
            )
            if self._can_merge_prefill_contents(all_batch_contents[-1],
                                                new_batch_contents):
                merge_contents(all_batch_contents[-1], new_batch_contents)
            else:
                all_batch_contents.append(new_batch_contents)
        return all_batch_contents

    def _make_attn_bias(self, context_groups, token_groups):
        dtype = self.dtype
        is_causal = True  # TODO: add support for non-causal tasks
        context_groups = torch.tensor(context_groups,
                                      device='cpu',
                                      dtype=torch.int16)
        context_groups = context_groups.repeat_interleave(self.block_size,
                                                          dim=-1)
        context_len = context_groups.size(-1)
        token_groups = torch.tensor(token_groups,
                                    device='cpu',
                                    dtype=torch.int16)
        num_queries = token_groups.size(-1)
        seq_groups = torch.cat([context_groups, token_groups], dim=-1)
        attn_mask = seq_groups.unflatten(-1,
                                         (1, -1)) != token_groups.unflatten(
                                             -1, (-1, 1))
        if is_causal:
            causal_mask = torch.ones(num_queries,
                                     num_queries,
                                     device='cpu',
                                     dtype=torch.bool)
            causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0)
            attn_mask[:, :, context_len:].logical_or_(causal_mask)
        attn_mask = attn_mask.to(dtype).masked_fill_(attn_mask, -math.inf)

        return attn_mask.unflatten(0, (1, -1))

    def _form_prefill_batch(self, contents):
        if len(contents.req_ids) == 0:
            return PrefillInputData()

        token_ids = contents.token_ids
        req_ids = contents.req_ids
        query_lens = [len(tids) for tids in contents.token_ids]
        if self.profiler.enabled:
            self.profiler_counter_helper.capture_prompt_seq_stats(query_lens)
        context_lens = contents.context_lens

        token_positions = [
            list(range(cl, cl + ql))
            for cl, ql in zip(context_lens, query_lens)
        ]
        block_assignment = [[
            divmod(pos, self.block_size) for pos in positions
        ] for positions in token_positions]
        token_slots = [[
            blocks[bi] * self.block_size + bo for bi, bo in assignment
        ] for blocks, assignment in zip(contents.blocks, block_assignment)]
        token_groups = [[i] * len(tid) for i, tid in enumerate(token_ids)]
        num_context_blocks = [
            round_up(ctx_len, self.block_size) // self.block_size
            for ctx_len in context_lens
        ]
        context_blocks: list = [
            blocks[:num]
            for blocks, num in zip(contents.blocks, num_context_blocks)
        ]
        num_context_blocks = [len(b) for b in context_blocks]
        context_groups = [[i] * b for i, b in enumerate(num_context_blocks)]
        has_context = sum(context_lens) > 0

        target_bs, target_seq, target_blocks = self._get_prompt_bucketing_fn()(
            query_lens, num_context_blocks)
        token_ids = self._align_and_pad(contents.token_ids,
                                        (target_bs, target_seq),
                                        itertools.repeat(-1))
        token_positions = self._align_and_pad(token_positions,
                                              (target_bs, target_seq),
                                              itertools.repeat(-1))
        token_slots = self._align_and_pad(token_slots, (target_bs, target_seq),
                                          itertools.repeat(-1))
        token_groups = self._align_and_pad(token_groups,
                                           (target_bs, target_seq),
                                           itertools.repeat(-1))
        context_blocks = self._align_and_pad(context_blocks,
                                             (target_bs, target_blocks),
                                             itertools.repeat(-1))
        context_groups = self._align_and_pad(context_groups,
                                             (target_bs, target_blocks),
                                             itertools.repeat(-1))

        # TODO: cycle through dummy slots and blocks
        #dummy_slots = itertools.cycle(
        #    range(self._PAD_SLOT_ID, self._PAD_SLOT_ID + self.block_size))

        cur_offset = 0
        logits_indices = []
        logits_requests = []
        for req_id, qlen, log_pos in zip(req_ids, query_lens,
                                         contents.logits_positions):
            source = [cur_offset + x for x in log_pos]
            dest = [req_id] * len(log_pos)
            logits_indices.extend(source)
            logits_requests.extend(dest)
            if self.use_merged_prefill:
                cur_offset += qlen
            else:
                cur_offset += len(token_ids[0])

        attn_bias = None
        if self.use_merged_prefill:
            attn_bias = self._make_attn_bias(context_groups, token_groups)
            attn_bias = attn_bias.to('hpu', non_blocking=True)
        else:
            attn_bias = None

        logits_indices = pad_list(
            logits_indices,
            round_up(len(logits_indices), self.logits_rounding),
            itertools.repeat(-1))

        query_lens = _async_h2d_tensor(query_lens, torch.int32)
        token_ids = _async_h2d_tensor(token_ids, torch.int32)
        token_positions = _async_h2d_tensor(token_positions, torch.int32)
        token_slots = _async_h2d_tensor(token_slots, torch.int64)
        logits_indices = _async_h2d_tensor(logits_indices, torch.int32)
        context_lens = _async_h2d_tensor(context_lens, torch.int32)
        context_blocks_t: Optional[torch.tensor]
        if has_context:
            context_blocks_t = _async_h2d_tensor(context_blocks,
                                                 torch.int32).flatten()
        else:
            context_blocks_t = None

        attn_metadata = HPUAttentionMetadataV1.make_prefill_metadata(
            seq_lens_tensor=query_lens,
            context_lens_tensor=context_lens,
            slot_mapping=token_slots,
            block_list=context_blocks_t,
            attn_bias=attn_bias,
            block_size=self.block_size)

        return PrefillInputData(request_ids=[req_ids],
                                prompt_lens=[query_lens],
                                token_ids=[token_ids],
                                position_ids=[token_positions],
                                attn_metadata=[attn_metadata],
                                logits_indices=[logits_indices],
                                logits_requests=[logits_requests])

    def _prepare_prefill_inputs(
            self, num_prefills, num_decodes,
            num_scheduled_tokens: list[int]) -> PrefillInputData:

        all_batch_contents = self._extract_prefill_batch_contents(
            num_prefills, num_decodes, num_scheduled_tokens)
        all_batches = [
            self._form_prefill_batch(bc) for bc in all_batch_contents
        ]
        merge_contents(all_batches[0], *all_batches[1:])
        return all_batches[0]

    def _prepare_decode_inputs(self, num_decodes,
                               num_scheduled_tokens) -> DecodeInputData:
        # Decodes run as one single padded batch with shape [batch, 1]
        #
        # We need to set _PAD_SLOT_ID for the padding tokens in the
        # slot_mapping, such that the attention KV cache insertion
        # logic knows to ignore those indicies. Otherwise, the
        # padding data can be dummy since we have a causal mask.

        block_table_cpu_tensor = self.input_batch.block_table[
            0].get_cpu_tensor()
        if num_decodes == 0:
            return DecodeInputData(num_decodes=0)
        # BLOCK_TABLE [batch, max_num_blocks_per_req]
        context_lens = self.input_batch.num_computed_tokens_cpu[:num_decodes]

        # NOTE(kzawora): the +1 is what causes this entire thing to work,
        # as in the paged attention, we don't fetch just the context from cache,
        # but also kvs for the current token
        num_blocks = np.ceil(
            (context_lens + 1) / self.block_size).astype(np.int32).tolist()

        # PAD FOR STATIC SHAPES.
        padded_batch_size: int
        padded_batch_size = self.bucketing_manager.find_decode_bucket(
            num_decodes, sum(num_blocks))[0]

        block_tables_list = []
        for i, n in enumerate(num_blocks):
            seq_block_table = block_table_cpu_tensor[i, :n].tolist()
            assert len(seq_block_table) == n
            block_tables_list.append(seq_block_table)

        # POSITIONS. [batch, 1]
        # We slice at the end, since we use the positions for gathering.
        positions = torch.zeros((padded_batch_size, 1), dtype=torch.int32)
        positions[:num_decodes] = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu.reshape(-1,
                                                             1)[:num_decodes])
        positions = positions[:padded_batch_size]

        padded_index = torch.zeros((padded_batch_size, 1), dtype=torch.int64)
        index = positions.to(torch.int64)[:num_decodes]
        padded_index[:num_decodes] = index

        # TOKEN_IDS. [batch, 1]
        token_ids = torch.zeros((padded_batch_size, 1), dtype=torch.int32)
        token_ids[:num_decodes] = torch.gather(input=torch.from_numpy(
            self.input_batch.token_ids_cpu),
                                               dim=1,
                                               index=index)

        # SLOT_MAPPING [batch, 1]
        # The "slot" is the "physical index" of a token in the KV cache.
        # Look up the block_idx in the block table (logical<>physical map)
        # to compute this.
        block_number = torch.ones(
            (padded_batch_size, 1), dtype=torch.int32) * self._PAD_BLOCK_ID
        block_number[:num_decodes] = torch.gather(input=block_table_cpu_tensor,
                                                  dim=1,
                                                  index=(index //
                                                         self.block_size))
        block_offsets = padded_index % self.block_size
        slot_mapping = block_number * self.block_size + block_offsets
        # set an out of range value for the padding tokens so that they
        # are ignored when inserting into the KV cache.
        slot_mapping = slot_mapping[:padded_batch_size]
        dummy_slots = itertools.cycle(
            range(self._PAD_SLOT_ID, self._PAD_SLOT_ID + self.block_size))
        slot_mapping[num_decodes:].apply_(lambda _, ds=dummy_slots: next(ds))

        # CONTEXT_LENS [batch_size]
        block_list, block_groups, block_usage = \
            self.get_habana_paged_attn_buffers(
            block_tables_list, slot_mapping.tolist(), padded_batch_size)

        logits_indices = torch.zeros(padded_batch_size,
                                     dtype=torch.int32,
                                     device='cpu')
        query_start_loc = torch.empty((num_decodes + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens[:num_decodes],
                  out=query_start_loc_np[1:])
        logits_indices[:num_decodes] = query_start_loc[1:] - 1
        num_decode_tokens = torch.tensor(np.sum(context_lens), device='cpu')

        # CPU<>HPU sync *should not* happen here.
        token_ids_device = _async_h2d_tensor_copy(token_ids, self.device)
        positions_device = _async_h2d_tensor_copy(positions, self.device)
        logits_indices_device = _async_h2d_tensor_copy(logits_indices,
                                                       self.device)
        block_list_device = _async_h2d_tensor_copy(block_list, self.device)
        block_usage_device = _async_h2d_tensor_copy(block_usage, self.device)
        block_groups_device = _async_h2d_tensor_copy(block_groups, self.device)
        num_decode_tokens_device = _async_h2d_tensor_copy(
            num_decode_tokens, self.device)
        slot_mapping_device = _async_h2d_tensor_copy(slot_mapping, self.device)
        return DecodeInputData(
            num_decodes=num_decodes,
            token_ids=token_ids_device,
            position_ids=positions_device,
            logits_indices=logits_indices_device,
            attn_metadata=HPUAttentionMetadataV1.make_decode_metadata(
                block_list=block_list_device,
                block_usage=block_usage_device,
                block_groups=block_groups_device,
                input_positions=None,
                num_decode_tokens=num_decode_tokens_device,
                slot_mapping=slot_mapping_device,
                block_size=self.block_size,
            ))

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_prefills,
        num_decodes,
    ) -> tuple[PrefillInputData, Optional[DecodeInputData]]:

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0

        num_reqs = num_prefills + num_decodes

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        num_prompt_tokens = []
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            assert req_id is not None
            seq_num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            seq_num_prompt_tokens = self.input_batch.num_prompt_tokens[idx]
            num_scheduled_tokens.append(seq_num_scheduled_tokens)
            num_prompt_tokens.append(seq_num_prompt_tokens)
            # NOTE: assert that all the decodes are "decodes".
            if idx < num_decodes:
                assert seq_num_scheduled_tokens == 1
        return (self._prepare_prefill_inputs(num_prefills, num_decodes,
                                             num_scheduled_tokens),
                self._prepare_decode_inputs(num_decodes, num_scheduled_tokens))

    def _seq_len(self, attn_metadata):
        return attn_metadata.slot_mapping.size(-1)

    def _num_blocks(self, attn_metadata):
        if attn_metadata.block_list is None:
            return 0
        return attn_metadata.block_list.numel()

    def _check_config(self, batch_size, seq_len, num_blocks, attn_metadata,
                      warmup_mode):
        phase = "prompt" if attn_metadata.is_prompt else "decode"
        cfg = (batch_size, seq_len, num_blocks, phase)
        seen = cfg in self.seen_configs
        self.seen_configs.add(cfg)
        if not seen and not warmup_mode:
            logger.warning(
                "Configuration: (%s, %s, %s, %s) was not warmed-up!", phase,
                batch_size, seq_len, num_blocks)

    def _execute_model_generic(self,
                               token_ids,
                               position_ids,
                               attn_metadata,
                               logits_indices,
                               kv_caches,
                               warmup_mode=False):

        # FORWARD.
        batch_size = token_ids.size(0)
        seq_len = self._seq_len(attn_metadata)
        num_blocks = self._num_blocks(attn_metadata)
        self._check_config(batch_size, seq_len, num_blocks, attn_metadata,
                           warmup_mode)
        additional_kwargs = {}
        if htorch.utils.internal.is_lazy(
        ) and not self.model_config.enforce_eager:
            use_graphs = self._use_graphs()
            additional_kwargs.update({"bypass_hpu_graphs": not use_graphs})
        else:
            # no hpu graphs for t.compile?
            use_graphs = False
        trimmed_attn_metadata = trim_attn_metadata(attn_metadata)
        if self.is_driver_worker:
            model_event_name = ("model_forward_"
                                f"bs{batch_size}_"
                                f"seq{seq_len}_"
                                f"ctx{num_blocks}_"
                                f"graphs{'T' if use_graphs else 'F'}")
        else:
            model_event_name = 'model_executable'
        with self.profiler.record_event('internal', model_event_name):
            hidden_states = self.model.forward(
                input_ids=token_ids,
                positions=position_ids,
                attn_metadata=trimmed_attn_metadata,
                kv_caches=kv_caches)
        # NOTE(kzawora): returning hidden_states is required in prompt logprobs
        # scenarios, as they will do logit processing on their own
        non_flattened_hidden_states = hidden_states

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = hidden_states[logits_indices]
        with self.profiler.record_event('internal', ('compute_logits'
                                                     f'{batch_size}_'
                                                     f'seq{seq_len}_ctx'
                                                     f'{num_blocks}')):
            logits = self.model.compute_logits(hidden_states, None)
        return non_flattened_hidden_states, logits

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for i, (req_id, num_prompt_logprobs) in enumerate(
                num_prompt_logprobs_dict.items()):

            num_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Determine number of logits to retrieve.
            start_tok = request.num_computed_tokens + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens < num_remaining_tokens:
                # This is a chunk, more tokens remain.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            prompt_hidden_states = hidden_states[i, :num_logits]
            logits = self.model.compute_logits(prompt_hidden_states, None)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer GPU->CPU async.
            prompt_logprobs_dict[req_id] = LogprobsTensors(
                token_ids.to("cpu", non_blocking=True),
                logprobs.to("cpu", non_blocking=True),
                ranks.to("cpu", non_blocking=True),
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        torch.hpu.synchronize()

        return prompt_logprobs_dict

    def _is_quant_with_inc(self):
        quant_config = os.getenv("QUANT_CONFIG", None) is not None
        return (self.model_config.quantization == "inc" or quant_config)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        # NOTE(kzawora): Since scheduler doesn't differentiate between prefills
        # and decodes, we must handle mixed batches. In _update_states we make
        # sure that first self.input_batch.num_decodes requests are decodes,
        # and remaining ones until the end are prefills. _update_states also
        # handles changes in request cache based on scheduler outputs and
        # previous iterations (e.g. keeping block tables and context lengths up
        # to date, creating, pruning and updating request caches,
        # and some more stuff)

        # If num_decodes == self.input_batch.num_reqs, then batch is all decode, and only a single decode forward pass will be executed in this method. # noqa
        # If num_decodes == 0, then batch is all prefill, and only prefill forward passes will be executed  in this method. # noqa
        # If neither apply, then batch is mixed, and both prefill and decode forward passes will be executed in this method. # noqa

        # First, we will execute all decodes (if any) in a single batch,
        # then we'll execute prefills in batches of up to max_prefill_batch_size elements. # noqa
        # All shapes used in forward passes are bucketed appropriately to mitigate risk of graph recompilations. # noqa

        # We perform sampling directly after executing each forward pass
        # Everything is done asynchronously - the only sync point is the place
        # where we copy the generated tokens back to the host.

        # Example: If a batch has 6 requests, 3 prefills and 3 decodes, the unprocessed sequences in batch will be laid as follows: # noqa
        # [D0, D1, D2, P0, P1, P2]
        # If we assume max_prefill_batch_size=2, the flow of this method will look as follows: # noqa
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1], [P2] (BS=2 + BS=1 bucket, no seqs padding) # noqa
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS2 [P0, P1]
        # prefill[iter 0] compute_logits BS2 [P0, P1]
        # prefill[iter 0] sampler BS2 [P0, P1] -> [tokP0, tokP1]
        # prefill[iter 1] forward pass BS1 [P0, P1]
        # prefill[iter 1] compute_logits BS1 [P0, P1]
        # prefill[iter 1] sampler BS1 [P0, P1] -> [tokP2]
        # prefill concat sampler results [tokP0, tokP1], [tokP2] -> [tokP0, tokP1, tokP2] # noqa
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] # noqa
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2] # noqa
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        # Example2: Same thing, but with max_prefill_batch_size=4:
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1, P2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS4 [P0, P1, P2, 0]
        # prefill[iter 0] compute_logits BS4 [P0, P1, P2, 0]
        # prefill[iter 0] sampler BS4 [P0, P1, P2, 0] -> [tokP0, tokP1, tokP2, 0] # noqa
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] # noqa
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2] # noqa
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        batch_changed = self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT
        # If necessary, swap decodes/prompts to have all decodes on the start
        ensure_decodes_first(self.input_batch)
        # Prepare prompts/decodes info
        pd_info = self._get_prompts_and_decodes(scheduler_output)
        num_decodes = len(pd_info.decode_req_ids)
        num_prefills = len(pd_info.prompt_req_ids)
        num_reqs = num_decodes + num_prefills
        with self.profiler.record_event('internal', 'prepare_input_tensors'):
            prefill_data, decode_data = self._prepare_inputs(
                scheduler_output, num_prefills, num_decodes)
        #FIXME(kzawora): Currently there's no handling of logprobs. Fix that
        # later.
        prefill_sampled_token_ids = []
        prefill_sampled_requests = []
        decode_sampled_token_ids = []
        decode_sampled_requests = []
        ######################### PREFILLS #########################
        if num_prefills > 0:
            htorch.core.mark_step()
            for idx, (req_id, prompt_len, token_ids, position_ids,
                      attn_metadata, logits_indices,
                      logits_requests) in enumerate(
                          zip(*shallow_tuple(prefill_data))):
                self.event_start = self.profiler.get_timestamp_us()
                self.profiler.start("internal", "prefill")
                htorch.core.mark_step()
                prefill_hidden_states_ts, logits_device = \
                    self._execute_model_generic(
                        token_ids, position_ids, attn_metadata, logits_indices,
                        self.kv_caches)
                htorch.core.mark_step()
                with self.profiler.record_event('internal', "sampler"):
                    sampling_metadata = self._prepare_sampling(
                        batch_changed, req_id, pad_to=logits_device.shape[0])
                    sampler_output = self.sampler(
                        logits=logits_device,
                        sampling_metadata=sampling_metadata)
                    prefill_sampled_token_ids.append(
                        sampler_output.sampled_token_ids.flatten())
                    prefill_sampled_requests.extend(logits_requests)
                htorch.core.mark_step()
                if self.is_driver_worker and self.profiler.enabled:
                    # Stop recording 'execute_model_generic' event
                    self.profiler.end()
                    event_end = self.profiler.get_timestamp_us()
                    counters = self.profiler_counter_helper.get_counter_dict(
                        cache_config=self.cache_config,
                        duration=event_end - self.event_start,
                        seq_len=self._seq_len(attn_metadata),
                        batch_size_padded=token_ids.size(0),
                        real_batch_size=len(req_id),
                        prompt_batch_idx=idx,
                        is_prompt=True)
                    self.profiler.record_counter(self.event_start, counters)
            if self.is_driver_worker and self.profiler.enabled:
                self.profiler_counter_helper.reset_prompt_seq_stats()

        ######################### DECODES #########################
        # Decodes run as one single batch with [padded_decode_bs, 1]
        if num_decodes > 0:
            self.event_start = self.profiler.get_timestamp_us()
            self.profiler.start("internal", "decode")
            assert decode_data is not None
            htorch.core.mark_step()
            _, logits_device = self._execute_model_generic(
                decode_data.token_ids, decode_data.position_ids,
                decode_data.attn_metadata, decode_data.logits_indices,
                self.kv_caches)
            htorch.core.mark_step()
            with self.profiler.record_event('internal', "sampler"):
                sampling_metadata = self._prepare_sampling(
                    batch_changed,
                    pd_info.decode_req_ids,
                    pad_to=logits_device.shape[0])
                sampler_output = self.sampler(
                    logits=logits_device, sampling_metadata=sampling_metadata)
                decode_sampled_token_ids.append(
                    sampler_output.sampled_token_ids.flatten())
                decode_sampled_requests.extend(
                    self.input_batch.req_ids[:num_decodes])
            htorch.core.mark_step()
            if self.is_driver_worker and self.profiler.enabled:
                # Stop recording 'execute_model' event
                self.profiler.end()
                event_end = self.profiler.get_timestamp_us()
                counters = self.profiler_counter_helper.get_counter_dict(
                    cache_config=self.cache_config,
                    duration=event_end - self.event_start,
                    seq_len=self._seq_len(decode_data.attn_metadata),
                    batch_size_padded= \
                        decode_data.token_ids.size(0),  # type: ignore
                    real_batch_size=decode_data.num_decodes,
                    prompt_batch_idx=None,
                    is_prompt=False)
                self.profiler.record_counter(self.event_start, counters)

        # From this point onward, all operations are done on CPU.
        # We already have tokens. Let's copy the data to
        # CPU as is, and then discard padded tokens.
        with self.profiler.record_event('internal', "sampler_postprocessing"):
            prefill_sampled_token_ids = [
                tensor.cpu() for tensor in prefill_sampled_token_ids
            ]
            decode_sampled_token_ids = [
                tensor.cpu()[:num_decodes]
                for tensor in decode_sampled_token_ids
            ]
            sampled_token_ids_list = torch.cat(
                decode_sampled_token_ids + prefill_sampled_token_ids).tolist()
            sampled_token_requests = \
                decode_sampled_requests + prefill_sampled_requests
            max_req_index = max(self.input_batch.req_id_to_index.values())
            postprocessed_sampled_token_ids: list[list]
            postprocessed_sampled_token_ids = [[]
                                               for _ in range(max_req_index +
                                                              1)]
            for tok_id, req_id in zip(sampled_token_ids_list,
                                      sampled_token_requests):
                postprocessed_sampled_token_ids[
                    self.input_batch.req_id_to_index[req_id]].append(tok_id)

        # NOTE(kzawora): idk what happens if part of batch doesn't have logprobs

        ######### UPDATE REQUEST STATE WITH GENERATED TOKENS #########
        for req_id in self.input_batch.req_ids[:num_reqs]:
            req_state = self.requests[req_id]
            i = self.input_batch.req_id_to_index[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            token_ids = postprocessed_sampled_token_ids[i]
            num_tokens = len(token_ids)
            self.input_batch.token_ids_cpu[i, seq_len:seq_len +
                                           num_tokens] = token_ids
            self.input_batch.num_tokens[i] += len(token_ids)
            req_state.output_token_ids.extend(token_ids)

        # NOTE(chendi): enable cache based on PR(#20291)
        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(
                postprocessed_sampled_token_ids[:num_reqs]):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)
        ################## RETURN ##################
        # Create output.
        all_req_ids = pd_info.decode_req_ids + pd_info.prompt_req_ids
        #prompt_logprobs_dict: dict[
        #    str, Optional[LogprobsTensors]] = self._get_prompt_logprobs_dict(
        #        prefill_hidden_states_device, scheduler_output)
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        all_req_ids = pd_info.decode_req_ids + pd_info.prompt_req_ids
        logprobs = None

        model_runner_output = ModelRunnerOutput(
            req_ids=all_req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=postprocessed_sampled_token_ids,
            logprobs=logprobs,
            spec_token_ids=None,
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            pooler_output=[],
        )
        return model_runner_output

    def load_model(self) -> None:
        import habana_frameworks.torch.core as htcore
        if self.model_config.quantization == 'inc' or \
            self.model_config.quantization == 'fp8':
            htcore.hpu_set_env()
        logger.info("Starting to load model %s...", self.model_config.model)
        with HabanaMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        if self._is_quant_with_inc():
            logger.info("Preparing model with INC..")
            with HabanaMemoryProfiler() as m_inc:
                from neural_compressor.torch.quantization import (FP8Config,
                                                                  convert,
                                                                  prepare)
                config = FP8Config.from_json_file(os.getenv(
                    "QUANT_CONFIG", ""))
                if config.measure:
                    self.model = prepare(self.model, config)
                elif config.quantize:
                    self.model = convert(self.model, config)
                else:
                    raise ValueError(
                        "Unknown quantization config mode,"
                        "please validate quantization config file")
                htcore.hpu_initialize(self.model,
                                      mark_only_scales_as_const=True)
            self.inc_initialized_successfully = True
            self.model_memory_usage = m_inc.consumed_device_memory
            logger.info("Preparing model with INC took %.4f GB",
                        self.model_memory_usage / float(2**30))
        elif not is_fake_hpu():
            self.model = self.model.to("hpu")
            htcore.mark_step()

        hidden_layer_markstep_interval = int(
            os.getenv('VLLM_CONFIG_HIDDEN_LAYERS', '1'))
        model_config = getattr(self.model, "config", None)
        modify_model_layers(
            self.model,
            get_target_layer_suffix_list(
                model_config.model_type if model_config is not None else None),
            hidden_layer_markstep_interval)
        torch.hpu.synchronize()

        with HabanaMemoryProfiler() as m:  # noqa: SIM117
            self.model = _maybe_wrap_in_hpu_graph(self.model,
                                                  vllm_config=self.vllm_config)
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Wrapping in HPUGraph took %.4f GB",
                    self.model_memory_usage / float(2**30))

        with HabanaMemoryProfiler() as m:
            self._maybe_compile(self.model)
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Compilation took %.4f GB",
                    self.model_memory_usage / float(2**30))

    def _maybe_compile(self, *args, **kwargs):
        if not is_fake_hpu() and not htorch.utils.internal.is_lazy(
        ) and not self.vllm_config.model_config.enforce_eager:
            if os.getenv('VLLM_REGIONAL_COMPILATION',
                         'true').strip().lower() in ("1", "true"):
                compiled_methods = [
                    '_update_metadata', '_rotary_prepare_cos_sin'
                ]
                for method_name in compiled_methods:
                    method = getattr(self.model, method_name)
                    if method is not None:
                        self._compile_region(self.model, method_name, method)
                self.regional_compilation_layers_list = [
                    RMSNorm, VocabParallelEmbedding
                ]
                self._regional_compilation(self.model)
            else:
                self.model = self._compile(self.model)

    def _regional_compilation(self,
                              module,
                              parent_module=None,
                              module_name=None):
        if isinstance(module, torch.nn.ModuleList):
            for children_name, children_module in module.named_children():
                self._compile_region(module, children_name, children_module)
        elif any(
                isinstance(module, layer)
                for layer in self.regional_compilation_layers_list):
            self._compile_region(
                parent_module,
                module_name,
                module,
            )
        else:
            for children_name, children_module in module.named_children():
                self._regional_compilation(children_module, module,
                                           children_name)

    def _compile_region(self, model, name, module):
        module = self._compile(module)
        setattr(model, name, module)

    def _compile(self, module):
        if not hasattr(self, '_compile_config'):
            fullgraph = os.getenv('VLLM_T_COMPILE_FULLGRAPH',
                                  'false').strip().lower() in ("1", "true")
            dynamic = os.getenv('VLLM_T_COMPILE_DYNAMIC_SHAPES',
                                'false').strip().lower() in ("1", "true")
            self._compile_config = {'fullgraph': fullgraph, 'dynamic': dynamic}
        fullgraph = self._compile_config['fullgraph']
        dynamic = self._compile_config['dynamic']
        if dynamic:
            return torch.compile(module,
                                 backend='hpu_backend',
                                 fullgraph=fullgraph,
                                 options={"force_static_compile": True})
        else:
            return torch.compile(module,
                                 backend='hpu_backend',
                                 fullgraph=fullgraph,
                                 dynamic=False)

    def _use_graphs(self):
        return not self.model_config.enforce_eager

    def log_graph_warmup_summary(self, buckets, is_prompt, total_mem):
        phase = f'Graph/{"Prompt" if is_prompt else "Decode"}'
        msg = (f'{phase} captured:{len(buckets)} '
               f'used_mem:{format_bytes(total_mem)}')
        logger.info(msg)

    def warmup_scenario(self,
                        batch_size,
                        seq_or_block,
                        num_blocks,
                        is_prompt,
                        kv_caches,
                        is_pt_profiler_run=True) -> None:
        """Dummy warmup run for memory usage and graph compilation."""

        query_seq_len = seq_or_block if is_prompt else 1
        input_ids = torch.zeros((batch_size, query_seq_len),
                                dtype=torch.int32,
                                device='cpu')
        position_ids = torch.zeros((batch_size, query_seq_len),
                                   dtype=torch.int32,
                                   device='cpu')
        slot_mapping = torch.zeros((batch_size, query_seq_len),
                                   dtype=torch.int64,
                                   device='cpu')

        input_ids_device = _async_h2d_tensor_copy(input_ids, self.device)
        position_ids_device = _async_h2d_tensor_copy(position_ids, self.device)
        slot_mapping_device = _async_h2d_tensor_copy(slot_mapping, self.device)

        use_graphs = self._use_graphs()
        phase = "prompt" if is_prompt else "decode"
        scenario_name = ("warmup_"
                         f"{phase}_"
                         f"bs{batch_size}_"
                         f"seq{query_seq_len}_"
                         f"ctx{num_blocks}_"
                         f"graphs{'T' if use_graphs else 'F'}")
        input_ids = torch.zeros((batch_size, query_seq_len),
                                dtype=torch.int32,
                                device='cpu')
        position_ids = torch.zeros((batch_size, query_seq_len),
                                   dtype=torch.int32,
                                   device='cpu')
        slot_mapping = torch.zeros((batch_size, query_seq_len),
                                   dtype=torch.int64,
                                   device='cpu')

        input_ids_device = _async_h2d_tensor_copy(input_ids, self.device)
        position_ids_device = _async_h2d_tensor_copy(position_ids, self.device)
        slot_mapping_device = _async_h2d_tensor_copy(slot_mapping, self.device)
        self.profiler.start('internal', scenario_name)

        times = 3 if use_graphs or is_pt_profiler_run else 1
        for time_index in range(times):
            if is_prompt:
                seq_lens = torch.zeros((batch_size),
                                       dtype=torch.int32,
                                       device='cpu')
                seq_lens.fill_(seq_or_block)
                seq_lens_device = _async_h2d_tensor_copy(seq_lens, self.device)
                block_list_device = None
                if num_blocks:
                    prefix_block_tables = torch.ones(
                        (batch_size, num_blocks),
                        dtype=torch.int32,
                        device='cpu') * self._PAD_BLOCK_ID
                    block_list_device = _async_h2d_tensor_copy(
                        prefix_block_tables.flatten(), self.device)
                attn_metadata = \
                    HPUAttentionMetadataV1.make_prefill_metadata(
                        attn_bias=None,
                        seq_lens_tensor=seq_lens_device,
                        context_lens_tensor=seq_lens_device,
                        slot_mapping=slot_mapping_device,
                        block_list=block_list_device,
                        block_size=self.block_size)
            else:
                block_tables = [
                    x.tolist()
                    for x in np.array_split(np.arange(num_blocks), batch_size)
                ]
                block_list, block_groups, block_usage = \
                    self.get_habana_paged_attn_buffers(
                        slot_mapping=slot_mapping,
                        block_tables=block_tables,
                        batch_size=batch_size)
                block_list_device = _async_h2d_tensor_copy(
                    block_list, self.device)
                block_usage_device = _async_h2d_tensor_copy(
                    block_usage, self.device)
                block_groups_device = _async_h2d_tensor_copy(
                    block_groups, self.device)
                attn_metadata = HPUAttentionMetadataV1.make_decode_metadata(
                    block_list=block_list_device,
                    block_usage=block_usage_device,
                    block_groups=block_groups_device,
                    num_decode_tokens=batch_size,
                    input_positions=None,
                    slot_mapping=slot_mapping_device,
                    block_size=self.block_size)

        logits_indices = torch.arange(0, batch_size, device='cpu')
        logits_indices_device = _async_h2d_tensor_copy(logits_indices,
                                                       self.device)
        # Dummy run.
        htorch.core.mark_step()
        _ = self._execute_model_generic(input_ids_device, position_ids_device,
                                        attn_metadata, logits_indices_device,
                                        kv_caches, True)
        # TODO: do sampling on logits, warmup sampler and prefill joiner
        htorch.core.mark_step()
        self.profiler.end()
        return None

    def log_warmup(self, phase, i, max_i, batch_size, seq_len, num_blocks):
        free_mem = format_bytes(
            HabanaMemoryProfiler.current_free_device_memory())
        msg = (f"[Warmup][{phase}][{i+1}/{max_i}] "
               f"batch_size:{batch_size} "
               f"query_len:{seq_len} "
               f"num_blocks:{num_blocks} "
               f"free_mem:{free_mem}")
        logger.info(msg)

    def warmup_graphs(self,
                      buckets,
                      is_prompt,
                      kv_caches,
                      starting_mem=0,
                      total_batch_seq=0.001):
        total_mem = starting_mem
        idx = 0
        num_candidates = len(buckets)
        captured_all = True
        for idx, (batch_size, seq_len,
                  num_blocks) in enumerate(reversed(buckets)):
            # Graph memory usage is proportional to seq dimension in a batch
            phase = f"Graph/{'prompt' if is_prompt else 'decode'}"
            if is_prompt:
                if num_blocks:
                    batch_seq = batch_size * seq_len * num_blocks
                else:
                    batch_seq = batch_size * seq_len
            else:
                batch_seq = batch_size

            graphed_bucket = (batch_size, seq_len, num_blocks, is_prompt)
            if graphed_bucket in self.graphed_buckets:
                continue
            self.graphed_buckets.add(graphed_bucket)
            self.log_warmup(phase, idx, num_candidates, batch_size, seq_len,
                            num_blocks)
            with HabanaMemoryProfiler() as mem_prof:
                self.warmup_scenario(batch_size, seq_len, num_blocks,
                                     is_prompt, kv_caches)
            #TODO(kzawora): align_workers
            used_mem = mem_prof.consumed_device_memory
            total_mem += used_mem
            total_batch_seq += batch_seq

        return total_mem, total_batch_seq, captured_all

    def _add_dummy_request(self, requests, num_scheduled_tokens,
                           num_computed_tokens, total_tokens,
                           scheduled_tokens):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData

        num_blocks = round_up(total_tokens, self.block_size) // self.block_size
        prompt_token_ids = list(range(total_tokens))

        req_id = f'req-{len(requests)}'
        block_ids = [0] * num_blocks
        sampling_params = SamplingParams(temperature=0.0)

        req = NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_inputs=[],
            mm_hashes=[],
            mm_positions=[],
            sampling_params=sampling_params,
            block_ids=[block_ids],
            num_computed_tokens=num_computed_tokens,
            lora_request=None,
        )
        requests.append(req)
        num_scheduled_tokens[req_id] = scheduled_tokens

    @staticmethod
    def _generate_seq_lengths(num_samples, num_blocks, block_size):
        assert num_samples <= num_blocks
        blocks = [num_blocks // num_samples] * num_samples
        missing_blocks = num_blocks - sum(blocks)
        for i in range(missing_blocks):
            blocks[i] += 1
        seq_lengths = [b * block_size - 1 for b in blocks]
        return seq_lengths

    def _execute_dummy_scenario(self, prompt_cfg, decode_cfg):
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
        requests: list[NewRequestData] = []
        scheduled_tokens: dict[str, int] = {}

        if prompt_cfg:
            prompt_bs, prompt_query_len, prompt_blocks = prompt_cfg
            prompt_ctx_len = prompt_blocks * self.block_size
            prompt_total_tokens = prompt_query_len + prompt_ctx_len
            for _ in range(prompt_bs):
                self._add_dummy_request(requests,
                                        scheduled_tokens,
                                        num_computed_tokens=prompt_ctx_len,
                                        total_tokens=prompt_total_tokens,
                                        scheduled_tokens=prompt_query_len)
        if decode_cfg:
            decode_bs, decode_blocks = decode_cfg
            decode_seq_lengths = self._generate_seq_lengths(
                decode_bs, decode_blocks, self.block_size)
            for dsl in decode_seq_lengths:
                self._add_dummy_request(requests,
                                        scheduled_tokens,
                                        num_computed_tokens=dsl,
                                        total_tokens=dsl,
                                        scheduled_tokens=1)
        sched_output = SchedulerOutput(
            scheduled_new_reqs=requests,
            scheduled_cached_reqs=[],
            num_scheduled_tokens=scheduled_tokens,
            total_num_scheduled_tokens=sum(scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        cleanup = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=[],
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(req.req_id for req in requests),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        self.execute_model(sched_output)
        self.execute_model(cleanup)

    def _generate_profiling(self, prompt_cfg, decode_cfg):
        steps = 3
        profiler = setup_profiler(warmup=steps - 1, active=1)
        torch.hpu.synchronize()
        profiler.start()
        for _ in range(steps):
            self._execute_dummy_scenario(prompt_cfg, decode_cfg)
            torch.hpu.synchronize()
            profiler.step()
        profiler.stop()

    @staticmethod
    def _parse_profile_cfg(profile_cfg):
        if profile_cfg:
            return tuple(map(int, profile_cfg.split(',')))
        return None

    @staticmethod
    def _parse_legacy_profile_cfg(profile_cfg):
        if profile_cfg:
            cfg = profile_cfg.split('_')
            assert cfg[0] in ['prompt', 'decode']
            return (cfg[0], int(cfg[1]), int(cfg[2]), cfg[3] == 't')
        return None

    def _read_profiling_cfg(self):
        prompt_cfg = self._parse_profile_cfg(
            os.environ.get('VLLM_PROFILE_PROMPT', None))
        decode_cfg = self._parse_profile_cfg(
            os.environ.get('VLLM_PROFILE_DECODE', None))
        legacy_cfg = self._parse_legacy_profile_cfg(
            os.environ.get('VLLM_PT_PROFILE', None))
        if legacy_cfg and not (prompt_cfg or decode_cfg):
            phase, bs, seq_or_blocks, use_graphs = legacy_cfg
            assert use_graphs != self.model_config.enforce_eager, \
                "'use_graphs' is out of sync with model config. " \
                "Either change the flag or change vllm engine parameters"
            if phase == 'prompt':
                prompt_cfg = (bs, seq_or_blocks, 0)
            else:
                decode_cfg = (bs, seq_or_blocks)
        return prompt_cfg, decode_cfg

    @torch.inference_mode()
    def warmup_model(self) -> None:
        if not self.enable_bucketing:
            return
        prompt_profile_cfg, decode_profile_cfg = self._read_profiling_cfg()
        if prompt_profile_cfg or decode_profile_cfg:
            self._generate_profiling(prompt_profile_cfg, decode_profile_cfg)
            raise AssertionError("Finished profiling")
        kv_caches = self.kv_caches
        self.bucketing_manager.generate_prompt_buckets()
        self.bucketing_manager.generate_decode_buckets()

        if not htorch.utils.internal.is_lazy(
        ) and not self.model_config.enforce_eager:
            multiplier = 3 if os.getenv('VLLM_REGIONAL_COMPILATION',
                                        'true').lower() in ('1', 'true') else 1
            cache_size_limit = 1 + multiplier * (
                len(self.bucketing_manager.prompt_buckets) +
                len(self.bucketing_manager.decode_buckets))
            torch._dynamo.config.cache_size_limit = max(
                cache_size_limit, torch._dynamo.config.cache_size_limit)
            # Multiply by 8 to follow the original default ratio between
            # the cache_size_limit and accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = max(
                cache_size_limit * 8,
                torch._dynamo.config.accumulated_cache_size_limit)

        if self.skip_warmup or self.use_merged_prefill:
            logger.info("Skipping warmup...")
            return

        self.profiler.start('internal', 'warmup')
        start_mem = HabanaMemoryProfiler.current_device_memory_usage()
        start_time = time.perf_counter()

        compile_only_mode_context = functools.partial(bc.env_setting,
                                                      "PT_COMPILE_ONLY_MODE",
                                                      True)
        can_use_compile_only_mode = True
        try:
            with compile_only_mode_context():
                pass
            logger.debug("Using PT_COMPILE_ONLY_MODE.")
        except KeyError:
            can_use_compile_only_mode = False
            logger.warning('Cannot use PT_COMPILE_ONLY_MODE. '
                           'Warmup time will be negatively impacted. '
                           'Please update Gaudi Software Suite.')
        with compile_only_mode_context(
        ) if can_use_compile_only_mode else contextlib.nullcontext():
            if not self.model_config.enforce_eager:
                assert self.mem_margin is not None, \
                    ("HabanaWorker.determine_num_available_blocks needs "
                    "to be called before warming up the model.")
                #TODO(kzawora): align_workers
                mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                    self.warmup_graphs(
                    self.bucketing_manager.prompt_buckets,
                    True, kv_caches)
                mem_post_decode, decode_batch_seq, decode_captured_all = \
                    self.warmup_graphs(
                    self.bucketing_manager.decode_buckets,
                    False, kv_caches)

                self.log_graph_warmup_summary(
                    self.bucketing_manager.prompt_buckets, True,
                    mem_post_prompt)
                self.log_graph_warmup_summary(
                    self.bucketing_manager.decode_buckets, False,
                    mem_post_decode)

        end_time = time.perf_counter()
        end_mem = HabanaMemoryProfiler.current_device_memory_usage()
        if os.getenv('VLLM_FULL_WARMUP',
                     'false').strip().lower() in ("1", "true"):
            # Since the model is warmed up for all possible tensor sizes,
            # Dynamo can skip checking the guards
            torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        elapsed_time = end_time - start_time
        msg = (
            f"Warmup finished in {elapsed_time:.0f} secs, "
            f"allocated {format_bytes(end_mem - start_mem)} of device memory")
        logger.info(msg)
        self.profiler.end()

    def shutdown_inc(self):
        can_finalize_inc = self._is_quant_with_inc() and \
            (self.model.model is not None) and \
            self.inc_initialized_successfully and \
            not self._is_inc_finalized
        if can_finalize_inc:
            from neural_compressor.torch.quantization import (
                finalize_calibration)
            finalize_calibration(self.model.model)
            self._is_inc_finalized = True

    def __del__(self):
        self.shutdown_inc()

    @torch.inference_mode()
    def profile_run(self) -> None:
        return
        """Profile to measure peak memory during forward pass."""

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value `None`.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers

        # Run empty prefill forwards - prefill max batch and prefill max seq
        self.warmup_scenario(batch_size=1,
                             seq_or_block=self.max_model_len,
                             is_prompt=True,
                             kv_caches=kv_caches)
        max_seq_len = math.ceil(
            (self.max_num_tokens // self.max_prefill_batch_size) /
            self.block_size) * self.block_size
        self.warmup_scenario(batch_size=self.max_prefill_batch_size,
                             seq_or_block=max_seq_len,
                             is_prompt=True,
                             kv_caches=kv_caches)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) == 0:
            return None
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: dict[str, torch.Tensor] = {}

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = \
                    kv_cache_tensor.size // kv_cache_spec.page_size_bytes
                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks + 1, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    v_cache_shape = None if self.model_config.use_mla \
                    else kv_cache_shape
                    dtype = kv_cache_spec.dtype
                    key_cache = torch.zeros(kv_cache_shape,
                                            dtype=dtype,
                                            device=self.device)
                    if v_cache_shape is not None:
                        value_cache = torch.zeros(v_cache_shape,
                                                  dtype=dtype,
                                                  device=self.device)
                    else:
                        value_cache = None
                    for layer_name in kv_cache_tensor.shared_by:
                        kv_caches[layer_name] = (key_cache, value_cache)
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")
            layer_names = set()
            for group in kv_cache_config.kv_cache_groups:
                layer_names.update(group.layer_names)
            assert layer_names == set(
                kv_caches.keys()), "Some layers are not correctly initialized"
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

        if self.enable_bucketing:
            self.bucketing_manager.num_hpu_blocks = num_blocks
        self._PAD_BLOCK_ID = num_blocks
        self._PAD_SLOT_ID = num_blocks * self.block_size

        htorch.hpu.synchronize()

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        return list(model.pooler.get_supported_tasks())

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def _get_nans_in_logits(
        self,
        logits: Optional[torch.Tensor],
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None
                    and req_index < logits.shape[0] else 0)
            return num_nans_in_logits
        except IndexError:
            return {}

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, \
                f"Config `{config_name}` not supported. " \
                f"Allowed configs: {allowed_config_names}"
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def reload_weights(self) -> None:
        assert getattr(self, "model", None) is not None, \
            "Cannot reload weights before model is loaded."
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model_loader.load_weights(self.model, model_config=self.model_config)
        torch.hpu.synchronize()

def get_path_to_rope(model: torch.nn.Module):
    """Dynamically get the path to the RotaryEmbedding layer in the model.
    This function will recursively search through the module hierarchy to find
    a RotaryEmbedding layer and return the full path to that layer as a list
    of names.
    If no such layer is found, it returns None.
    """

    def find_rope_layer(parent, path):
        # Base case: check if this parent is None
        if parent is None:
            return None

        # Check if the current layer is a RotaryEmbedding
        if hasattr(parent, 'named_children'):
            for child_name, child_module in parent.named_children():
                # If the current child is of type RotaryEmbedding,
                # return the full path
                if child_module.__class__.__name__.endswith("RotaryEmbedding"):
                    return path + [child_name]
                # Otherwise, recurse into this child to check its children
                result = find_rope_layer(child_module, path + [child_name])
                if result is not None:
                    return result
        return None

    # Start the search from the top level model
    path_to_rope = find_rope_layer(model, [])

    # Return the result if found, otherwise None
    return path_to_rope

class PrepareDecodeMetadata(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: List[List[int]]
    attn_metadata: Optional[AttentionMetadata]
    # lora_index_mapping: List[List[int]]
    # lora_prompt_mapping: List[List[int]]
    # lora_requests: Set[LoRARequest]
    slot_mapping: List[List[int]]
    # lora_ids: List[int]

    @classmethod
    def empty(cls):
        return PrepareDecodeMetadata(input_tokens=[],
                                     input_positions=[],
                                     attn_metadata=None,
                                    #  lora_index_mapping=[],
                                    #  lora_prompt_mapping=[],
                                    #  lora_requests=set(),
                                     slot_mapping=[],)
                                    #  lora_ids=[])
        
class PreparePromptMetadata(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: List[List[int]]
    attn_metadata: Optional[AttentionMetadata]
    seq_lens: List[int]
    query_lens: List[int]
    lora_index_mapping: List[List[int]]
    lora_prompt_mapping: List[List[int]]
    # lora_requests: set[LoRARequest]
    # multi_modal_kwargs: Optional[Dict[str, BatchedTensorInputs]]
    slot_mapping: List[List[int]]
    lora_ids: List[int]

    @classmethod
    def empty(cls):
        return PreparePromptMetadata(input_tokens=[],
                                     input_positions=[],
                                     attn_metadata=None,
                                     seq_lens=[],
                                     query_lens=[],
                                     lora_index_mapping=[],
                                     lora_prompt_mapping=[],
                                     lora_requests=set(),
                                     multi_modal_kwargs=None,
                                     slot_mapping=[],
                                     lora_ids=[])

def align_workers(value, op):
    group = get_world_group().cpu_group
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return value
    value_t = torch.tensor(value, device='cpu')
    torch.distributed.all_reduce(value_t, op=op, group=group)
    return value_t.item()
       
class HPUModelRunnerBase(ModelRunnerBase[TModelInputForHPU]):
    """
    Helper class for shared methods between GPU model runners.
    """
    _model_input_cls: type[TModelInputForHPU]

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        # mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        # environment.set_model_config(self.model_config)
        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.sliding_window = (self.model_config.get_sliding_window()
                               if self.model_config is not None else None)
        self.device_config = (self.device_config if self.device_config
                              is not None else DeviceConfig())
        if is_fake_hpu():
            self.device_config.device = torch.device('cpu')
            self.device_config.device_type = 'cpu'
            self.load_config.device = None
        self.device = self.device_config.device
        self.enforce_eager = self.model_config.enforce_eager
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.max_num_prefill_seqs = self.max_num_seqs 
            # if self.scheduler_config.max_num_prefill_seqs is not None \
            #     else self.max_num_seqs
        self.max_model_len = self.scheduler_config.max_model_len
        self.max_num_batched_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.block_size = self.cache_config.block_size

        self.pin_memory = is_pin_memory_available()
        self.kv_cache_dtype = self.cache_config.cache_dtype

        num_attn_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        needs_attn_backend = (num_attn_heads != 0
                              or self.model_config.is_attention_free)
        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        ) if needs_attn_backend else None

        # Multi-modal data support
        # self.input_registry = input_registry
        # self.mm_registry = mm_registry
        # self.mm_registry = MULTIMODAL_REGISTRY
        # self.multi_modal_input_mapper = self.mm_registry \
        #     .create_input_mapper(self.model_config)
        # self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization
        # self.lora_manager: LRUCacheWorkerLoRAManager = None
        # self.model: torch.nn.Module = None
        # self.inc_initialized_successfully = False

        # Profiler stats
        self.profiler = HabanaHighLevelProfiler()
        self.profiler_counter_helper = HabanaProfilerCounterHelper()
        self.seen_configs: set = set()
        self._mem_margin: Optional[int] = None
        # self.bucketing_ctx = linear.HPUBucketingContext(self.max_num_seqs,
        #                                          self.max_num_prefill_seqs,
        #                                          self.block_size,
        #                                          self.max_num_batched_tokens)
        self.graphed_buckets: set[Any] = set()

        # self._set_gc_threshold()
        self.use_contiguous_pa = os.environ.get('VLLM_CONTIGUOUS_PA',
                                                'true').lower() == 'true'
        if vllm_config.speculative_config is not None \
            and self.use_contiguous_pa:
            raise ValueError(
                "Speculative decoding is not supported with "
                "contiguous PA, please set VLLM_CONTIGUOUS_PA=false")
        # For multi-step scheduling
        self.cached_step_outputs: List[torch.Tensor] = []
        self.is_pooler = False

    # def _set_gc_threshold(self) -> None:
    #     # Read https://docs.python.org/3/library/gc.html#gc.set_threshold
    #     # for comprehensive description of gc generations.
    #     # We can either use VLLM_GC_THR_GEN[0-2] (this has higher priority)
    #     # to set particular generation threshold or use simpler
    #     # VLLM_GC_THR_MULTIPLIER to multiply default values.
    #     default_gc_thrs = list(gc.get_threshold())
    #     requested_gc_thrs = [0] * len(default_gc_thrs)
    #     for i in range(len(default_gc_thrs)):
    #         requested_gc_thrs[i] = int(
    #             os.environ.get(f'VLLM_GC_THR_GEN{i}', default_gc_thrs[i]))
    #     if requested_gc_thrs == default_gc_thrs:
    #         gc_thr_multiplier = int(os.environ.get('VLLM_GC_THR_MULTIPLIER',
    #                                                2))
    #         requested_gc_thrs = [
    #             t * gc_thr_multiplier for t in default_gc_thrs
    #         ]
    #     gc.set_threshold(*requested_gc_thrs)

        # Multi-modal data support
        # self.multi_modal_input_mapper = MULTIMODAL_REGISTRY \
        #     .create_input_mapper(self.model_config)

        # self.skip_warmup = os.environ.get('VLLM_SKIP_WARMUP',
        #                                   'false').lower() == 'true'

    def load_model(self) -> None:
        import habana_frameworks.torch.core as htcore
        if self.model_config.quantization == 'inc' or \
           self.model_config.quantization == 'fp8':
            htcore.hpu_set_env()
        with HabanaMemoryProfiler() as m:
            with HabanaMemoryProfiler() as m_getmodel:
                self.model = get_model(vllm_config=self.vllm_config)
            msg = ("Pre-loading model weights on "
                   f"{next(self.model.parameters()).device} "
                   f"took {m_getmodel.get_summary_string()}")
            logger.info(msg)
            self.is_pooler = hasattr(self.model, "_pooler")
            if self.lora_config:
                assert hasattr(self.model, "supported_lora_modules"
                               ) and self.model.supported_lora_modules, (
                                   "Model does not support LoRA")
                assert hasattr(self.model, "embedding_modules"
                               ), "Model does not have embedding_modules"
                assert hasattr(
                    self.model, "embedding_padding_modules"
                ), "Model does not have embedding_padding_modules"
                assert not self.lora_config.bias_enabled, \
                    "Bias support in LoRA is not enabled in HPU yet."
                assert not self.lora_config.fully_sharded_loras, \
                    "Fully sharded LoRAs is not enabled in HPU yet."
                # if supports_multimodal(self.model):
                #     logger.warning(
                #         "Regarding multimodal models, vLLM currently "
                #         "only supports adding LoRA to language model.")
                # It's necessary to distinguish between the
                # max_position_embeddings of VLMs and LLMs.
                if hasattr(self.model.config, "max_position_embeddings"):
                    max_pos_embeddings = (
                        self.model.config.max_position_embeddings)
                else:
                    max_pos_embeddings = (
                        self.model.config.text_config.max_position_embeddings)

                # self.lora_manager = LRUCacheWorkerLoRAManager(
                #     self.scheduler_config.max_num_seqs,
                #     self.scheduler_config.max_num_batched_tokens,
                #     self.vocab_size,
                #     self.lora_config,
                #     self.device,
                #     self.model.embedding_modules,
                #     self.model.embedding_padding_modules,
                #     max_position_embeddings=max_pos_embeddings,
                # )
                # self.model = self.lora_manager.create_lora_manager(self.model)

            if self.model_config.quantization == 'inc':
                logger.info("Preparing model with INC..")
                with HabanaMemoryProfiler() as m_inc:
                    from neural_compressor.torch.quantization import (
                        FP8Config, convert, prepare)
                    config = FP8Config.from_json_file(
                        os.getenv("QUANT_CONFIG", ""))
                    if config.measure:
                        self.model = prepare(self.model, config)
                    elif config.quantize:
                        self.model = convert(self.model, config)
                    htcore.hpu_initialize(self.model,
                                          mark_only_scales_as_const=True)
                self.inc_initialized_successfully = True
                logger.info("Preparing model with INC took %s",
                            m_inc.get_summary_string())
            elif not is_fake_hpu():
                self.model = self.model.to("hpu")
                htcore.mark_step()

            hidden_layer_markstep_interval = int(
                os.getenv('VLLM_CONFIG_HIDDEN_LAYERS', '1'))
            model_config = getattr(self.model, "config", None)
            modify_model_layers(
                self.model,
                get_target_layer_suffix_list(
                    model_config.
                    model_type if model_config is not None else None),
                hidden_layer_markstep_interval)
            path_to_rope = get_path_to_rope(self.model)
            torch.hpu.synchronize()

            with HabanaMemoryProfiler() as m_wrap:
                self.model = self._maybe_wrap_in_hpu_graph(
                    self.model,
                    vllm_config=self.vllm_config)
                    # layer_names=path_to_rope)
            msg = f"Wrapping in HPU Graph took {m_wrap.get_summary_string()}"
            logger.info(msg)

        self.model_memory_usage = m.consumed_device_memory
        msg = f"Loading model weights took in total {m.get_summary_string()}"
        logger.info(msg)

    def _add_dummy_seq(self, seq_group_metadata_list, is_prompt):
        real_batch_size = len(seq_group_metadata_list)
        batch_size_padded = self.bucketing_ctx.get_padded_batch_size(
            real_batch_size, is_prompt)
        batch_size_padding = batch_size_padded - real_batch_size

        seq_group_metadata_list = seq_group_metadata_list.copy()

        if batch_size_padding > 0:
            if self.is_pooler:
                temperature = None
            else:
                has_greedy_samples = any(
                    seq_group_metadata.sampling_params.temperature == 0.0
                    for seq_group_metadata in seq_group_metadata_list)
                temperature = 0.0 if has_greedy_samples else 1.0
            dummy_seq_group_metadata = self.create_dummy_seq_group_metadata(
                0, 0, is_prompt, temperature=temperature)
            seq_group_metadata_list.extend(dummy_seq_group_metadata
                                           for _ in range(batch_size_padding))
        return seq_group_metadata_list, real_batch_size, batch_size_padded

    def _maybe_wrap_in_hpu_graph(self, *args, **kwargs):
        return htorch.hpu.wrap_in_hpu_graph(
            HpuModelAdapter(*args, **kwargs), disable_tensor_cache=True
        ) if htorch.utils.internal.is_lazy() else HpuModelAdapter(
            *args, **kwargs)

    def get_model(self) -> nn.Module:
        if isinstance(self.model, HpuModelAdapter):
            return self.model.model
        return self.model

    def _use_graphs(self, batch_size, seq_len, is_prompt):
        if self.enforce_eager:
            return False
        if self.skip_warmup:
            return True
        return (batch_size, seq_len, is_prompt) in self.graphed_buckets

    def _is_valid_bucket(self, bucket):
        return bucket[0] * bucket[1] <= self.max_num_batched_tokens

    def _check_config(self, batch_size, seq_len, is_prompt, warmup_mode):
        cfg = (batch_size, seq_len, is_prompt)
        seen = cfg in self.seen_configs
        self.seen_configs.add(cfg)
        if not seen and not warmup_mode:
            phase = 'prompt' if is_prompt else 'decode'
            logger.warning("Configuration: (%s, %s, %s) was not warmed-up!",
                           phase, batch_size, seq_len)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> PreparePromptMetadata:
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        lora_index_mapping: List[List[int]] = []
        lora_prompt_mapping: List[List[int]] = []
        # lora_requests: set[LoRARequest] = set()

        seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        prefix_block_tables: List[List[int]] = []
        # multi_modal_kwargs_list: List[MultiModalKwargs] = []
        # multi_modal_placeholder_maps: Dict[
        #     str, MultiModalPlaceholderMap] = collections.defaultdict(
        #         MultiModalPlaceholderMap)

        if len(seq_group_metadata_list) == 0:
            return PreparePromptMetadata.empty()

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            computed_block_nums = seq_group_metadata.computed_block_nums
            if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None
                             or computed_block_nums == [])):
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching "
                    "now.")

            token_chunk_size = seq_group_metadata.token_chunk_size
            seq_data = seq_group_metadata.seq_data[seq_id]
            context_len = seq_data.get_num_computed_tokens()
            # We should use get_len here because in case of preemption
            # it contains output tokens.
            seq_len = min(seq_data.get_len(), context_len + token_chunk_size)
            prompt_tokens = seq_data.get_token_ids()[context_len:seq_len]
            seq_lens.append(seq_len)

            # NOTE: This only works for oooooooxxx style attention.
            if computed_block_nums is not None and len(
                    computed_block_nums) > 0 and self.sliding_window is None:
                # Prefix is not supported with sliding_window
                context_len = len(computed_block_nums) * self.block_size
                prompt_tokens = prompt_tokens[context_len:]
                prefix_block_tables.append(computed_block_nums)
            elif self.scheduler_config.chunked_prefill_enabled:
                if seq_group_metadata.block_tables is not None:
                    # Prefill has chunked before.
                    block_table = seq_group_metadata.block_tables[seq_id]
                    prefix_block_tables.append(block_table)
                else:
                    # The first prefill.
                    prefix_block_tables.append([])
            else:
                prefix_block_tables.append([])
                # Right now, prefill start is always 0. However, this
                # assumption can be changed once chunked prefill is introduced.
                assert context_len == 0

            # actual prompt lens
            context_lens.append(context_len)
            query_lens.append(seq_len - context_len)
            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(context_len, seq_len)))

            # if seq_group_metadata.multi_modal_data:
            #     positions = input_positions[0]
            #     mm_data, placeholder_maps = MultiModalPlaceholderMap \
            #         .from_seq_group(seq_group_metadata,
            #           range(positions[0], positions[0] + len(positions)))

            #     if self.mm_registry.has_processor(self.model_config):
            #         mm_kwargs = mm_data
            #     else:
            #         mm_kwargs = self.multi_modal_input_mapper(
            #             mm_data,
            #             seq_group_metadata.mm_processor_kwargs,
            #         )

            #     multi_modal_kwargs_list.append(mm_kwargs)

            #     for modality, placeholder_map in placeholder_maps.items():
            #         multi_modal_placeholder_maps[modality].extend(
            #             placeholder_map)

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([_PAD_SLOT_ID] * seq_len)
                continue

            # Compute the slot mapping.
            slot_mapping.append([])
            block_table = seq_group_metadata.block_tables[seq_id]

            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seq_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                assert context_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention")
                start_idx = max(0, seq_len - self.sliding_window)
            for i in range(context_len, seq_len):
                if i < start_idx:
                    slot_mapping[-1].append(_PAD_SLOT_ID)
                    continue
                # For encoder-only models, the block_table is None,
                # and there is no need to initialize the slot_mapping.
                if block_table is not None:
                    block_number = block_table[i // self.block_size]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slot_mapping[-1].append(slot)

        max_query_len = max(query_lens)
        real_num_seqs = len(query_lens)

        assert max_query_len > 0

        max_prompt_len = max(
            self.bucketing_ctx.get_padded_prompt_seq_len(max(seq_lens)),
            self.block_size)

        lora_ids: List[int] = []
        for seq_group_metadata, context_len in zip(seq_group_metadata_list,
                                                   context_lens):
            lora_id = seq_group_metadata.lora_int_id
            lora_ids.append(lora_id)

            # if lora_id > 0:
            #     lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * max_prompt_len
            lora_prompt_mapping.extend(
                [lora_id] *
                (max_prompt_len if seq_group_metadata.sampling_params and
                 seq_group_metadata.sampling_params.prompt_logprobs else 1))

        if any(context_lens):
            assert not self.scheduler_config.chunked_prefill_enabled
            # prefix caching

            max_num_block = max(len(bt) for bt in prefix_block_tables)
            prefix_block_list = list(
                itertools.chain.from_iterable(
                    bt if len(bt) == max_num_block else bt +
                    ([_PAD_BLOCK_ID] * (max_num_block - len(bt)))
                    for bt in prefix_block_tables))

            # TODO: pad to proper len
            pad_len = len(prefix_block_list)
            prefix_block_list = pad_list(prefix_block_list, pad_len,
                                         _PAD_BLOCK_ID)

            prefix_block_list_tensor = torch.tensor(prefix_block_list,
                                                    dtype=torch.long,
                                                    device='cpu')
        else:
            prefix_block_list_tensor = None

        input_tokens_tensor = make_tensor_with_pad(input_tokens,
                                                   max_len=max_prompt_len,
                                                   pad=0,
                                                   dtype=torch.long,
                                                   device='cpu')

        input_positions = make_tensor_with_pad(input_positions,
                                               max_len=max_prompt_len,
                                               pad=0,
                                               dtype=torch.long,
                                               device='cpu')

        slot_mapping = make_tensor_with_pad(slot_mapping,
                                            max_len=max_prompt_len,
                                            pad=_PAD_SLOT_ID,
                                            dtype=torch.long,
                                            device='cpu')
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.long,
                                       device='cpu')

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.long,
                                           device='cpu')

        # placeholder_index_maps = {
        #     modality: placeholder_map.index_map()
        #     for modality, placeholder_map in
        #     multi_modal_placeholder_maps.items()
        # }

        # Note: num_prefill_tokens is calculated using the length of
        # input_tokens after padding.
        num_prefill_tokens = input_tokens_tensor.numel()
        if prefix_block_list_tensor is not None:
            prefix_block_list_tensor = prefix_block_list_tensor.to(
                self.device, non_blocking=True)
        input_tokens_tensor = input_tokens_tensor.to(  # type: ignore
            self.device, non_blocking=True)
        input_positions = input_positions.to(  # type: ignore
            self.device, non_blocking=True)
        slot_mapping = slot_mapping.to(  # type: ignore
            self.device, non_blocking=True)
        seq_lens_tensor = seq_lens_tensor.to(self.device, non_blocking=True)
        context_lens_tensor = context_lens_tensor.to(self.device,
                                                     non_blocking=True)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            block_list=prefix_block_list_tensor,
            block_mapping=None,
            block_usage=None,
            block_indices=None,
            block_offsets=None,
            block_scales=None,
            block_groups=None,
            attn_bias=None,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            context_lens_tensor=context_lens_tensor,
            num_prefills=real_num_seqs,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            # multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
        )
        # multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)
        # for t in multi_modal_kwargs:
        #     if torch.is_tensor(multi_modal_kwargs[t]):
        #         multi_modal_kwargs[t] = multi_modal_kwargs[t].to(
        #             self.device, non_blocking=True)

        return PreparePromptMetadata(input_tokens=input_tokens_tensor,
                                     input_positions=input_positions,
                                     attn_metadata=attn_metadata,
                                     seq_lens=seq_lens,
                                     query_lens=query_lens,
                                     lora_index_mapping=lora_index_mapping,
                                     lora_prompt_mapping=lora_prompt_mapping,
                                    #  lora_requests=lora_requests,
                                    #  multi_modal_kwargs=multi_modal_kwargs,
                                     slot_mapping=slot_mapping,
                                     lora_ids=lora_ids)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        output=None,
    ) -> PrepareDecodeMetadata:
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        seq_lens: List[int] = []
        encoder_seq_lens: List[int] = []
        cross_block_tables: List[List[int]] = []
        block_tables: List[List[int]] = []
        lora_index_mapping: List[List[int]] = []
        lora_prompt_mapping: List[List[int]] = []
        # lora_requests: Set[LoRARequest] = set()

        is_enc_dec_model = self.model_config.is_encoder_decoder
        if len(seq_group_metadata_list) == 0:
            return PrepareDecodeMetadata.empty()
        lora_ids: List[int] = []

        dummy_slots = itertools.cycle(
            range(_PAD_SLOT_ID, _PAD_SLOT_ID + self.block_size))

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())
            lora_id = seq_group_metadata.lora_int_id
            lora_ids.append(lora_id)
            if is_enc_dec_model:
                for _ in range(len(seq_group_metadata.seq_data)):
                    encoder_seq_len = (
                        seq_group_metadata.encoder_seq_data.get_len()
                        if seq_group_metadata.encoder_seq_data else 0)
                    encoder_seq_lens.append(encoder_seq_len)
                    cross_block_table = seq_group_metadata.cross_block_table
                    cross_block_tables.append([] if (
                        cross_block_table is None) else cross_block_table)

            # if lora_id > 0:
            #     lora_requests.add(seq_group_metadata.lora_request)

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                if output is None:
                    generation_token = seq_data.get_last_token_id()
                    input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                num_fully_occupied_blocks = position // self.block_size
                block_table = block_table[:num_fully_occupied_blocks + 1]

                if len(block_table) == 0:
                    block_number = _PAD_BLOCK_ID
                else:
                    block_number = block_table[position // self.block_size]
                if block_number == _PAD_BLOCK_ID:
                    slot = next(dummy_slots)
                else:
                    block_offset = position % self.block_size
                    slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])
                lora_index_mapping.append(lora_id)
                lora_prompt_mapping.append(lora_id)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        if output is None:
            input_tokens = torch.tensor(input_tokens,
                                        dtype=torch.long,
                                        device='cpu')
        else:
            real_batch_size = len(seq_group_metadata_list)
            input_tokens = output[:real_batch_size].clone()

        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device='cpu')

        num_decode_tokens = len(seq_lens)

        last_block_usage = [
            slot[0] % self.block_size + 1 for slot in slot_mapping
        ]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
        block_usage = [[self.block_size] * (len(bt) - 1) + [lbu]
                       for bt, lbu in zip(block_tables, last_block_usage)
                       if bt]

        block_list = flatten(block_tables)
        block_groups = flatten(block_groups)
        block_usage = flatten(block_usage)

        assert len(block_list) == len(block_groups)
        assert len(block_list) == len(block_usage)

        if is_enc_dec_model:
            last_cross_block_usage = [
                (encoder_seq_len - 1) % self.block_size + 1
                for encoder_seq_len in encoder_seq_lens
            ]
            cross_block_groups = [[i] * len(bt)
                                  for i, bt in enumerate(cross_block_tables)]
            cross_block_usage = [
                [self.block_size] * (len(bt) - 1) + [lbu]
                for bt, lbu in zip(cross_block_tables, last_cross_block_usage)
                if bt
            ]
            cross_block_list = flatten(cross_block_tables)
            cross_block_groups = flatten(cross_block_groups)
            cross_block_usage = flatten(cross_block_usage)
            assert len(cross_block_list) == len(cross_block_groups)
            assert len(cross_block_list) == len(cross_block_usage)

        else:
            cross_block_list = None
            cross_block_groups = None
            cross_block_usage = None
            encoder_seq_lens_tensor = None

        padding_fn = None
        if self.use_contiguous_pa:
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            block_bucket_size = self.bucketing_ctx.get_padded_decode_num_blocks(
                block_bucket_size)
            indices: List[Any]
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(
                tensor, indices, pad_value)
        else:
            block_bucket_size = self.bucketing_ctx.get_padded_decode_num_blocks(
                len(block_list))
            padding_fn = lambda tensor, pad_value: pad_list(
                tensor, block_bucket_size, pad_value)

        block_list = padding_fn(block_list, _PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        if is_enc_dec_model:
            if self.use_contiguous_pa:
                cross_block_bucket_size = max(
                    max(cross_block_list) +
                    1, len(cross_block_list)) if cross_block_list else 0
                cross_block_bucket_size = \
                    self.bucketing_ctx.get_padded_decode_num_blocks(
                    cross_block_bucket_size)
                indices = [None] * cross_block_bucket_size
                for i, bid in enumerate(cross_block_list):
                    indices[bid] = i
                padding_fn = lambda tensor, pad_value: gather_list(
                    tensor, indices, pad_value)
            else:
                cross_block_bucket_size = \
                    self.bucketing_ctx.get_padded_decode_num_blocks(
                    len(cross_block_list))
                padding_fn = lambda tensor, pad_value: pad_list(
                    tensor, cross_block_bucket_size, pad_value)

            real_batch_size = len(seq_group_metadata_list)
            batch_size_padded = self.bucketing_ctx.get_padded_batch_size(
                real_batch_size, False)
            batch_size_padding = batch_size_padded - real_batch_size
            if batch_size_padding > 0:
                encoder_seq_lens.extend(encoder_seq_lens[0]
                                        for _ in range(batch_size_padding))
            cross_block_list = padding_fn(cross_block_list, _PAD_BLOCK_ID)
            cross_block_groups = padding_fn(cross_block_groups, -1)
            cross_block_usage = padding_fn(cross_block_usage, 1)

            cross_block_list = torch.tensor(cross_block_list,
                                            dtype=torch.int,
                                            device='cpu')
            cross_block_groups = torch.tensor(cross_block_groups,
                                              dtype=torch.int,
                                              device='cpu')
            cross_block_usage = torch.tensor(cross_block_usage,
                                             dtype=self.model_config.dtype,
                                             device='cpu')
            encoder_seq_lens_tensor = torch.tensor(encoder_seq_lens,
                                                   dtype=torch.long,
                                                   device='cpu')

        block_list = torch.tensor(block_list, dtype=torch.int, device='cpu')
        block_groups = torch.tensor(block_groups,
                                    dtype=torch.int,
                                    device='cpu')
        block_usage = torch.tensor(block_usage,
                                   dtype=self.model_config.dtype,
                                   device='cpu')
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device='cpu')

        input_tokens = input_tokens.to(  # type: ignore
            self.device, non_blocking=True)
        input_positions = input_positions.to(  # type: ignore
            self.device, non_blocking=True)
        block_list = block_list.to(  # type: ignore
            self.device, non_blocking=True)
        block_groups = block_groups.to(  # type: ignore
            self.device, non_blocking=True)
        block_usage = block_usage.to(  # type: ignore
            self.device, non_blocking=True)
        slot_mapping = slot_mapping.to(  # type: ignore
            self.device, non_blocking=True)
        if is_enc_dec_model:
            cross_block_list = cross_block_list.to(  # type: ignore
                self.device, non_blocking=True)
            cross_block_groups = cross_block_groups.to(  # type: ignore
                self.device, non_blocking=True)
            cross_block_usage = cross_block_usage.to(  # type: ignore
                self.device, non_blocking=True)
            encoder_seq_lens_tensor = encoder_seq_lens_tensor.to(  # type: ignore
                self.device, non_blocking=True)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            block_list=block_list,
            block_mapping=None,
            block_usage=block_usage,
            block_indices=None,
            block_offsets=None,
            block_scales=None,
            block_groups=block_groups,
            attn_bias=None,
            seq_lens_tensor=None,
            encoder_seq_lens=encoder_seq_lens,
            encoder_seq_lens_tensor=encoder_seq_lens_tensor,
            cross_block_list=cross_block_list,
            cross_block_groups=cross_block_groups,
            cross_block_usage=cross_block_usage,
            context_lens_tensor=None,
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )
        return PrepareDecodeMetadata(input_tokens=input_tokens,
                                     input_positions=input_positions,
                                     attn_metadata=attn_metadata,
                                     lora_index_mapping=lora_index_mapping,
                                     lora_prompt_mapping=lora_prompt_mapping,
                                    #  lora_requests=lora_requests,
                                     slot_mapping=slot_mapping,
                                     lora_ids=lora_ids)

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> tuple[TModelInputForHPU, SamplingMetadata]:
        if len(seq_group_metadata_list) == 0:
            return self._model_input_cls(), None

        input_tokens = None
        input_positions = None
        lora_mapping = None
        lora_requests = None
        multi_modal_kwargs = None
        batch_type = None
        seq_lens = None
        query_lens = None
        real_batch_size = None
        batch_size_padded = None

        self.event_start = self.profiler.get_timestamp_us()
        is_prompt = seq_group_metadata_list[0].is_prompt
        base_event_name = 'prompt' if is_prompt else 'decode'
        self.profiler.start('internal', base_event_name)

        seq_group_metadata_list, real_batch_size, batch_size_padded = (
            self._add_dummy_seq(seq_group_metadata_list, is_prompt))

        prefill_reqs = []
        decode_reqs = []
        for seq_group_meta in seq_group_metadata_list:
            if seq_group_meta.is_prompt:
                prefill_reqs.append(seq_group_meta)
            else:
                decode_reqs.append(seq_group_meta)

        # Prepare input tensors.
        (
            input_tokens,
            input_positions,
            prefill_attn_metadata,
            seq_lens,
            query_lens,
            lora_index_mapping,
            lora_prompt_mapping,
            lora_requests,
            multi_modal_kwargs,
            slot_mapping,
            lora_ids,
        ) = self._prepare_prompt(prefill_reqs)
        (
            decode_input_tokens,
            decode_input_positions,
            decode_attn_metadata,
            decode_lora_index_mapping,
            decode_lora_prompt_mapping,
            decode_lora_requests,
            decode_slot_mapping,
            decode_lora_ids,
        ) = self._prepare_decode(decode_reqs)

        if not self.is_pooler:
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, seq_lens, query_lens, self.device,
                self.pin_memory)

        if not self.scheduler_config.chunked_prefill_enabled:
            assert (len(prefill_reqs) and len(decode_reqs)) == 0

        num_prefills = len(seq_lens)
        num_prefill_tokens = len(input_tokens)
        num_decode_tokens = len(decode_input_tokens)

        # NOTE(kzawora): Here we diverge from GPU code - we don't
        # support mixed batches, so we either use decode or prefill
        # inputs, without coalescing.
        assert (num_prefills == 0 and num_decode_tokens > 0) or (
            num_prefills > 0
            and num_decode_tokens == 0), "HPU does not support mixed batches!"
        if num_decode_tokens > 0:
            input_tokens = decode_input_tokens
            input_positions = decode_input_positions
            slot_mapping = decode_slot_mapping
            lora_index_mapping = decode_lora_index_mapping
            lora_prompt_mapping = decode_lora_prompt_mapping
            lora_requests = decode_lora_requests
            lora_ids = decode_lora_ids

        # FIXME: We need to adjust selected_token_indices to accommodate
        # for padding
        max_len = input_tokens.size(1)
        paddings = [max_len - q for q in query_lens]
        paddings = [0] + paddings[:-1]
        paddings = list(itertools.accumulate(paddings))
        paddings_prompt_logprobs = []

        if not self.is_pooler:
            for i, seq_group_metadata in enumerate(seq_group_metadata_list):
                if seq_group_metadata.sampling_params \
                    and seq_group_metadata.sampling_params.prompt_logprobs \
                        is not None and seq_group_metadata.is_prompt:
                    paddings_prompt_logprobs += ([paddings[i]] * seq_lens[i])

            paddings = torch.tensor(
                paddings_prompt_logprobs
                if paddings_prompt_logprobs else paddings,
                dtype=sampling_metadata.selected_token_indices.dtype,
                device=sampling_metadata.selected_token_indices.device)
            sampling_metadata.selected_token_indices.add_(paddings)
        else:
            sampling_metadata = None

        # if self.lora_config:
        #     lora_mapping = LoRAMapping(
        #         **dict(index_mapping=lora_index_mapping,
        #                prompt_mapping=lora_prompt_mapping,
        #                is_prefill=(num_prefills > 0)))
        # else:
        lora_mapping = None

        if (prefill_attn_metadata is not None
                and decode_attn_metadata is not None):
            batch_type = BatchType.MIXED
            raise NotImplementedError("Mixed batch is not supported on HPU")
        elif prefill_attn_metadata is not None:
            batch_type = BatchType.PREFILL
        else:
            batch_type = BatchType.DECODE

        metadata_dict = {
            "input_tokens":
            input_tokens,
            "input_positions":
            input_positions,
            "selected_token_indices":
            sampling_metadata.selected_token_indices
            if sampling_metadata else None,
            "lora_requests":
            lora_requests,
            "lora_mapping":
            lora_mapping,
            "multi_modal_kwargs":
            multi_modal_kwargs,
            "num_prefill_tokens":
            num_prefill_tokens,
            "num_decode_tokens":
            num_decode_tokens,
            "slot_mapping":
            slot_mapping,
            "num_prefills":
            num_prefills,
            "batch_type":
            batch_type,
            "seq_lens":
            seq_lens,
            "query_lens":
            query_lens
        }
        if prefill_attn_metadata is not None:
            metadata_dict.update(prefill_attn_metadata.asdict_zerocopy())
        else:
            assert decode_attn_metadata is not None
            metadata_dict.update(decode_attn_metadata.asdict_zerocopy())

        attn_metadata = prefill_attn_metadata if \
            prefill_attn_metadata is not None else decode_attn_metadata

        return self._model_input_cls(input_tokens=input_tokens,
                                     seq_lens=seq_lens,
                                     query_lens=query_lens,
                                     input_positions=input_positions,
                                     attn_metadata=attn_metadata,
                                     lora_requests=lora_requests,
                                     lora_mapping=lora_mapping,
                                     multi_modal_kwargs=multi_modal_kwargs,
                                     real_batch_size=real_batch_size,
                                     batch_size_padded=batch_size_padded,
                                     lora_ids=lora_ids), \
                                        sampling_metadata

    def _seq_len(self, attn_metadata):
        if attn_metadata.num_prefills != 0:
            return attn_metadata.slot_mapping.size(1)
        else:
            return attn_metadata.block_list.numel()

    def trim_attn_metadata(self, metadata: AttentionMetadata) -> object:
        # NOTE(kzawora): To anyone working on this in the future:
        # Trimming metadata is required when using HPUGraphs.
        # Attention metadata is going to be hashed by PT bridge, and
        # appropriate HPUGraphs will be matched based on all inputs' hash.

        # Before you put more keys in here, make sure you know their
        # value type and make sure you know how it's going to be hashed.
        # You can find that information in input_hash function
        # in habana_frameworks/torch/hpu/graphs.py. You can also hash
        # it manually with torch.hpu.graphs.input_hash(attention_metadata)

        # If you use primitive types here - they will get hashed based
        # on their value. You *will* get lots of excessive graph captures
        # (and an OOM eventually) if you decide to put something like
        # seq_len int here.
        # If you absolutely need a scalar, put it in a tensor. Tensors
        # get hashed using their metadata, not their values:
        # input_hash(torch.tensor(123)) == input_hash(torch.tensor(321))
        # input_hash(123) != input_hash(321)
        # input_hash("abc") != input_hash("cba")
        attention_metadata = subtuple(metadata, 'TrimmedAttentionMetadata', [
            'attn_bias',
            'seq_lens_tensor',
            'context_lens_tensor',
            'block_list',
            'block_mapping',
            'block_usage',
            'slot_mapping',
            'is_prompt',
            'block_indices',
            'block_offsets',
            'block_scales',
            'block_groups',
        ])
        return attention_metadata

    def create_dummy_seq_group_metadata(self,
                                        group_id,
                                        seq_len,
                                        is_prompt,
                                        lora_request=None,
                                        temperature=0):
        if self.is_pooler:
            sampling_params = None
        # else:
        #     sampling_params = SamplingParams(temperature=temperature)
        #     num_blocks = math.ceil(seq_len / self.block_size)
        # seq_len = max(seq_len, 1)
        if is_prompt:
            input_len = seq_len
            output_len = 0
            block_tables = None
        else:
            input_len = seq_len - 1
            output_len = 1
            # block_tables = {group_id: [_PAD_BLOCK_ID] * num_blocks}
        prompt_token_ids = [0] * input_len
        output_token_ids = [1] * output_len
        prompt_token_ids_array = array('l', prompt_token_ids)  # noqa: F821
        seq_data = SequenceData(prompt_token_ids_array)
        seq_data.output_token_ids = output_token_ids
        return SequenceGroupMetadata(request_id=str(group_id),
                                     is_prompt=(output_len == 0),
                                     seq_data={group_id: seq_data},
                                     sampling_params=sampling_params,
                                     block_tables=block_tables,
                                     lora_request=lora_request)

    def profile_run(self) -> None:
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        bind_kv_cache(
            self.vllm_config.compilation_config.static_forward_context,
            [kv_caches])
        _, max_seq_len = self.bucketing_ctx.get_max_prompt_shape()
        max_batch_size = min(self.max_num_seqs,
                             self.max_num_batched_tokens // max_seq_len)

        self.warmup_scenario(max_batch_size, max_seq_len, True, kv_caches,
                             False, True)
        return

    def warmup_scenario(self,
                        batch_size,
                        seq_len,
                        is_prompt,
                        kv_caches,
                        is_pt_profiler_run=False,
                        is_lora_profile_run=False,
                        temperature=0) -> None:
        use_graphs = self._use_graphs(batch_size, seq_len, is_prompt)
        scenario_name = ("warmup_"
                         f"{'prompt' if is_prompt else 'decode'}_"
                         f"bs{batch_size}_"
                         f"seq{seq_len}_"
                         f"graphs{'T' if use_graphs else 'F'}")
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        # dummy_lora_requests: List[LoRARequest] = []
        # dummy_lora_requests_per_seq: List[LoRARequest] = []
        # if self.lora_config and is_lora_profile_run:
        #     assert self.lora_manager is not None
        #     with self.lora_manager.dummy_lora_cache():
        #         for idx in range(self.lora_config.max_loras):
        #             lora_id = idx + 1
        #             dummy_lora_request = LoRARequest(
        #                 lora_name=f"warmup_{lora_id}",
        #                 lora_int_id=lora_id,
        #                 lora_local_path="/not/a/real/path",
        #             )
        #             self.lora_manager.add_dummy_lora(dummy_lora_request,
        #                                              rank=LORA_WARMUP_RANK)
        #             dummy_lora_requests.append(dummy_lora_request)
        #         dummy_lora_requests_per_seq = [
        #             dummy_lora_requests[idx % len(dummy_lora_requests)]
        #             for idx in range(batch_size)
        #         ]
        self.profiler.start('internal', scenario_name)
        times = 3 if use_graphs or is_pt_profiler_run else 1
        if is_prompt:
            seqs = [
                self.create_dummy_seq_group_metadata(
                    i,
                    seq_len,
                    is_prompt,
                    # lora_request=dummy_lora_requests_per_seq[i]
                    # if dummy_lora_requests_per_seq else None,
                    temperature=temperature) for i in range(batch_size)
            ]
        else:
            # FIXME: seq_len is actually number of blocks
            blocks = [seq_len // batch_size for _ in range(batch_size)]
            blocks[0] += seq_len % batch_size
            seqs = [
                self.create_dummy_seq_group_metadata(
                    i,
                    b * self.block_size - 1,
                    is_prompt,
                    # lora_request=dummy_lora_requests_per_seq[i]
                    # if dummy_lora_requests_per_seq else None,
                    temperature=temperature) for i, b in enumerate(blocks)
            ]
        torch.hpu.synchronize()
        profiler = None
        if is_pt_profiler_run and self.is_driver_worker:
            profiler = setup_profiler()
            profiler.start()
        for _ in range(times):
            inputs = self.prepare_model_input(seqs)
            is_single_step = \
                self.vllm_config.scheduler_config.num_scheduler_steps == 1
            if is_prompt or is_single_step:
                self.execute_model(inputs, kv_caches, warmup_mode=True)
            else:  # decode with multi-step
                inputs = dataclass.replace(inputs,
                                             is_first_multi_step=True,
                                             is_last_step=False)
                self.execute_model(inputs,
                                   kv_caches,
                                   warmup_mode=True,
                                   num_steps=2,
                                   seqs=seqs)
                inputs = dataclass.replace(inputs,
                                             is_first_multi_step=False,
                                             is_last_step=True)
                self.execute_model(inputs,
                                   kv_caches,
                                   warmup_mode=True,
                                   num_steps=2,
                                   seqs=seqs)
            torch.hpu.synchronize()
            if profiler:
                profiler.step()
        if profiler:
            profiler.stop()
        self.profiler.end()
        gc.collect()

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_adapters()

    # def set_active_loras(self, lora_requests: Set[LoRARequest],
    #                      lora_mapping: LoRAMapping) -> None:
    #     if not self.lora_manager:
    #         raise RuntimeError("LoRA is not enabled.")
    #     self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    # def add_lora(self, lora_request: LoRARequest) -> bool:
    #     if not self.lora_manager:
    #         raise RuntimeError("LoRA is not enabled.")
    #     return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_adapters()

    def log_warmup(self, phase, i, max_i, batch_size, seq_len):
        free_mem = format_bytes(
            HabanaMemoryProfiler.current_free_device_memory())
        dim = "num_blocks"
        if "Prompt" in phase:
            dim = "seq_len"
        msg = (f"[Warmup][{phase}][{i+1}/{max_i}] "
               f"batch_size:{batch_size} "
               f"{dim}:{seq_len} "
               f"free_mem:{free_mem}")
        logger.info(msg)

    def warmup_all_buckets(self, buckets, is_prompt, kv_caches):
        for i, (batch_size, seq_len) in enumerate(reversed(buckets)):
            self.log_warmup('Prompt' if is_prompt else 'Decode', i,
                            len(buckets), batch_size, seq_len)
            self.warmup_scenario(batch_size, seq_len, is_prompt, kv_caches)

    def warmup_graphs(self,
                      strategy,
                      buckets,
                      is_prompt,
                      kv_caches,
                      available_mem,
                      starting_mem=0,
                      total_batch_seq=0.001):
        total_mem = starting_mem
        idx = 0
        phase = f'Graph/{"Prompt" if is_prompt else "Decode"}'
        num_candidates = len(buckets)
        ordering : Union[Callable[[Any], tuple[Any, Any]], \
            Callable[[Any], tuple[Any, Any, Any]]]
        if strategy == 'min_tokens':
            ordering = lambda b: (b[0] * b[1], b[1], b[0])
        elif strategy == 'max_bs':
            ordering = lambda b: (-b[0], b[1])
        else:
            raise NotImplementedError(
                f'Unsupported graph allocation strategy: {strategy}')
        buckets = list(sorted(buckets, key=ordering))
        captured_all = True
        warmed_random_sampler_bs: set[int] = set()
        for idx, (batch_size, seq_len) in enumerate(buckets):
            # Graph memory usage is proportional to seq dimension in a batch
            batch_seq = batch_size * seq_len if is_prompt else batch_size
            mem_estimate = batch_seq / total_batch_seq * total_mem
            if mem_estimate >= available_mem:
                captured_all = False
                continue
            graphed_bucket = (batch_size, seq_len, is_prompt)
            if graphed_bucket in self.graphed_buckets:
                continue
            self.graphed_buckets.add(graphed_bucket)
            self.log_warmup(phase, idx, num_candidates, batch_size, seq_len)
            with HabanaMemoryProfiler() as mem_prof:
                self.warmup_scenario(batch_size,
                                     seq_len,
                                     is_prompt,
                                     kv_caches,
                                     temperature=1.0 if batch_size
                                     not in warmed_random_sampler_bs else 0)
            warmed_random_sampler_bs.add(batch_size)
            used_mem = align_workers(mem_prof.consumed_device_memory,
                                     torch.distributed.ReduceOp.MAX)
            available_mem -= used_mem
            total_mem += used_mem
            total_batch_seq += batch_seq

        return total_mem, total_batch_seq, captured_all

    def log_graph_warmup_summary(self, buckets, is_prompt, total_mem):
        num_candidates = len(buckets)
        phase = f'Graph/{"Prompt" if is_prompt else "Decode"}'
        graphed = list(c[:2] for c in self.graphed_buckets
                       if c[2] == is_prompt)
        if num_candidates == 0:
            num_candidates = 1
        msg = (f'{phase} captured:{len(graphed)} '
               f'({100 * len(graphed) / num_candidates:.1f}%) '
               f'used_mem:{format_bytes(total_mem)} '
               f'buckets:{sorted(list(graphed))}')
        logger.info(msg)

    @torch.inference_mode()
    def warmup_model(self, kv_caches: List[torch.Tensor]) -> None:
        if profile := os.environ.get('VLLM_PT_PROFILE', None):
            phase, bs, seq_len, graph = profile.split('_')
            is_prompt = phase == 'prompt'
            graphs = graph == 't'
            if graphs:
                self.graphed_buckets.add((int(bs), int(seq_len), is_prompt))
            self.warmup_scenario(int(bs), int(seq_len), is_prompt, kv_caches,
                                 True)
            raise AssertionError("Finished profiling")

        self.bucketing_ctx.generate_prompt_buckets()
        if not self.is_pooler:
            max_blocks = kv_caches[0][0].size(0)
            self.bucketing_ctx.generate_decode_buckets(max_blocks)
        if not htorch.utils.internal.is_lazy() and not self.enforce_eager:
            multiplier = 3 if os.getenv('VLLM_REGIONAL_COMPILATION',
                                        'true').lower() == 'true' else 1
            cache_size_limit = 1 + multiplier * (
                len(self.bucketing_ctx.prompt_buckets) +
                len(self.bucketing_ctx.decode_buckets))
            torch._dynamo.config.cache_size_limit = max(
                cache_size_limit, torch._dynamo.config.cache_size_limit)
            # Multiply by 8 to follow the original default ratio between
            # the cache_size_limit and accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = max(
                cache_size_limit * 8,
                torch._dynamo.config.accumulated_cache_size_limit)
        if self.skip_warmup:
            logger.info("Skipping warmup...")
            return
        self.profiler.start('internal', 'warmup')
        start_mem = HabanaMemoryProfiler.current_device_memory_usage()
        start_time = time.perf_counter()

        compile_only_mode_context = functools.partial(bc.env_setting,
                                                      "PT_COMPILE_ONLY_MODE",
                                                      True)
        can_use_compile_only_mode = True
        try:
            with compile_only_mode_context():
                pass
            logger.debug("Using PT_COMPILE_ONLY_MODE.")
        except KeyError:
            can_use_compile_only_mode = False
            logger.warning('Cannot use PT_COMPILE_ONLY_MODE. '
                           'Warmup time will be negatively impacted. '
                           'Please update Gaudi Software Suite.')
        with compile_only_mode_context(
        ) if can_use_compile_only_mode else contextlib.nullcontext():
            self.warmup_all_buckets(self.bucketing_ctx.prompt_buckets, True,
                                    kv_caches)
            if not self.is_pooler:
                self.warmup_all_buckets(self.bucketing_ctx.decode_buckets,
                                        False, kv_caches)

            if not self.enforce_eager and htorch.utils.internal.is_lazy():
                if not self.is_pooler:
                    assert self.mem_margin is not None, \
                        ("HabanaWorker.determine_num_available_blocks needs "
                        "to be called before warming up the model.")

                free_mem = HabanaMemoryProfiler.current_free_device_memory()
                graph_free_mem = free_mem - self.mem_margin
                graph_free_mem = align_workers(graph_free_mem,
                                               torch.distributed.ReduceOp.MIN)
                prompt_strategy = os.environ.get('VLLM_GRAPH_PROMPT_STRATEGY',
                                                 'min_tokens')
                if not self.is_pooler:
                    prompt_graph_mem_ratio = float(
                        os.environ.get('VLLM_GRAPH_PROMPT_RATIO', '0.3'))
                    prompt_available_memory = (prompt_graph_mem_ratio *
                                               graph_free_mem)
                    decode_available_memory = (graph_free_mem -
                                               prompt_available_memory)
                    msg = (
                        f"Using {format_bytes(graph_free_mem)}"
                        f"/{format_bytes(free_mem)} "
                        "of free device memory for HPUGraphs, "
                        f"{format_bytes(prompt_available_memory)} \
                            for prompt and "
                        f"{format_bytes(decode_available_memory)} for decode "
                        f"(VLLM_GRAPH_PROMPT_RATIO={prompt_graph_mem_ratio})")
                    logger.info(msg)
                    mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                        self.warmup_graphs(
                        prompt_strategy, self.bucketing_ctx.prompt_buckets,
                        True, kv_caches, prompt_available_memory)

                    decode_strategy = os.environ.get(
                        'VLLM_GRAPH_DECODE_STRATEGY', 'max_bs')
                    mem_post_decode, decode_batch_seq, decode_captured_all = \
                        self.warmup_graphs(
                        decode_strategy, self.bucketing_ctx.decode_buckets,
                        False, kv_caches, decode_available_memory)

                    # Not all prompt buckets were captured, but all decode
                    # buckets were captured and we have some free
                    # graph-allocated space left. Let's try to use it for
                    # capturing more prompt buckets.
                    if (mem_post_decode + mem_post_prompt < graph_free_mem
                            and not prompt_captured_all
                            and decode_captured_all):
                        mem_post_prompt, _, prompt_captured_all = (
                            self.warmup_graphs(
                                prompt_strategy,
                                self.bucketing_ctx.prompt_buckets, True,
                                kv_caches, graph_free_mem - mem_post_prompt -
                                mem_post_decode, mem_post_prompt,
                                prompt_batch_seq))
                        # Not all decode buckets were captured, but all prompt
                        # buckets were captured and we have some free
                        # graph-allocated space left. Let's try to use it for
                        # capturing more decode buckets.
                        if mem_post_decode + mem_post_prompt < graph_free_mem \
                            and not decode_captured_all \
                                and prompt_captured_all:
                            mem_post_decode, _, _ = self.warmup_graphs(
                                decode_strategy,
                                self.bucketing_ctx.decode_buckets, False,
                                kv_caches, graph_free_mem - mem_post_prompt -
                                mem_post_decode, mem_post_decode,
                                decode_batch_seq)
                else:
                    prompt_available_memory = graph_free_mem
                    msg = (
                        f"Using {format_bytes(graph_free_mem)}"
                        f"/{format_bytes(free_mem)} "
                        "of free device memory for HPUGraphs, "
                        f"{format_bytes(prompt_available_memory)} for prompt")
                    logger.info(msg)
                    prompt_strategy = os.environ.get(
                        'VLLM_GRAPH_PROMPT_STRATEGY', 'min_tokens')

                    mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                        self.warmup_graphs(
                        prompt_strategy, self.bucketing_ctx.prompt_buckets,
                        True, kv_caches, prompt_available_memory)
                    if mem_post_prompt < graph_free_mem \
                        and not prompt_captured_all:
                        mem_post_prompt, _, prompt_captured_all = (
                            self.warmup_graphs(
                                prompt_strategy,
                                self.bucketing_ctx.prompt_buckets, True,
                                kv_caches, graph_free_mem - mem_post_prompt,
                                mem_post_prompt, prompt_batch_seq))

                self.log_graph_warmup_summary(
                    self.bucketing_ctx.prompt_buckets, True, mem_post_prompt)
                if not self.is_pooler:
                    self.log_graph_warmup_summary(
                        self.bucketing_ctx.decode_buckets, False,
                        mem_post_decode)

        end_time = time.perf_counter()
        end_mem = HabanaMemoryProfiler.current_device_memory_usage()
        elapsed_time = end_time - start_time
        msg = (
            f"Warmup finished in {elapsed_time:.0f} secs, "
            f"allocated {format_bytes(end_mem - start_mem)} of device memory")
        logger.info(msg)
        self.profiler.end()

    def finish_measurements(self):
        from neural_compressor.torch.quantization import finalize_calibration
        finalize_calibration(self.model.model)

    def shutdown_inc(self):
        can_finalize_inc = (self.model_config.quantization == 'inc') and \
            (self.model.model is not None) and \
            self.inc_initialized_successfully and \
            not getattr(self, "_is_inc_finalized", False)
        if can_finalize_inc:
            from neural_compressor.torch.quantization import (
                finalize_calibration)
            finalize_calibration(self.model.model)
            self._is_inc_finalized = True

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    @property
    def mem_margin(self) -> Optional[int]:
        return self._mem_margin

    @mem_margin.setter
    def mem_margin(self, value):
        self._mem_margin = value