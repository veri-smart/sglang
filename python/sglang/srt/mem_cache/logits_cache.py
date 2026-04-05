from __future__ import annotations
import heapq
from queue import Queue, Empty
import threading
import logging
import time
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Set, Iterator, Union, Callable, DefaultDict
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from dataclasses import dataclass, field
import torch
from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, AgentReqToTokenPool
from sglang.srt.server_args import get_global_server_args, ServerArgs
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.configs.model_config import ModelConfig
if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput


logger = logging.getLogger(__name__)


@dataclass
class Recorder:
    _input_tok: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=torch.int32))
    _input_kv: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=torch.int32))

    @staticmethod
    def _to_tensor(data: List | torch.Tensor) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.int32) if isinstance(data, List) else data

    @property
    def input_kv(self) -> torch.Tensor:
        return self._to_tensor(self._input_kv)

    @property
    def input_tok(self) -> torch.Tensor:
        return self._to_tensor(self._input_tok)


@dataclass
class LogitsRecord:
    req_to_token_pool: ReqToTokenPool
    server_args: ServerArgs
    tree_cache: BasePrefixCache
    # runtime states
    req_logits_key: Dict[str, List[LogitsKey]] = field(default_factory=dict)
    logits_cache: Dict[str, LogitsCache] = field(default_factory=dict)
    req_info: Dict[str, Recorder] = field(default_factory=dict)
    # mark a request's logits kv cache is already loaded
    req_already_loaded: Set[str] = field(default_factory=set)

    _sampler: Optional[object] = field(default=None, repr=False)
    model_config: Optional[ModelConfig] = field(default=None, repr=False)
    # used as async pre-compute logits cache
    producer_queue: Queue[Req] = field(default_factory=Queue, repr=False)
    consumer_queue: Dict[str, Queue[Tuple[List[int], Optional[torch.Tensor]]]] = field(
        default_factory=dict, repr=False)
    event_pool: Dict[str, threading.Event] = field(
        default_factory=dict, repr=False)

    def __post_init__(self):
        self.model_config = ModelConfig.from_server_args(self.server_args)

        t = threading.Thread(
            target=self.produce_nxt_logits,
            name="precompute-worker",
            daemon=True,
        )
        t.start()

    @property
    def sampler(self):
        if self._sampler is None:
            from sglang.srt.layers.sampler import Sampler
            self._sampler = Sampler()
        return self._sampler

    def sample_child(self, sampling_info: SamplingBatchInfo, seq_len, logits: Union[torch.Tensor, LogitsProcessorOutput]) -> int:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        if isinstance(logits, torch.Tensor):
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            elif logits.dim() >= 2:
                if logits.shape[0] != 1:
                    logits = logits[-1:].contiguous()

            cur_logit = LogitsProcessorOutput(next_token_logits=logits)
        else:
            cur_logit = logits
        nxt_id = self.sampler(
            logits_output=cur_logit,
            sampling_info=sampling_info,
            return_logprob=False,
            top_logprobs_nums=[],
            token_ids_logprobs=[],
            positions=seq_len,
            is_cpu=True,
        )
        return nxt_id

    def record_batch(self, batch: ScheduleBatch, logits_info: GenerationBatchResult):
        logits = logits_info.logits_output.cloned_next_token_logits
        logits_results = logits_info.next_token_ids
        for ind, req in enumerate(batch.reqs):
            if not req.should_cache:
                continue
            if req.rid in self.req_logits_key:
                self.req_logits_key[req.rid].append(
                    LogitsKey(logits[ind], logits_results[ind]))
                continue
            if batch.forward_mode.is_decode():
                # forget cached request
                self.req_logits_key[req.rid] = [
                    LogitsKey(logits[ind], logits_results[ind])]
                continue

            assert batch.forward_mode.is_extend(), "encounter strange mode"
            ins_kv_indices = batch.req_to_token_pool.req_to_token[req.req_pool_idx][: len(
                req.fill_ids)]
            self.req_logits_key[req.rid] = [
                LogitsKey(logits[ind], logits_results[ind])]
            self.req_info[req.rid] = Recorder(
                req.origin_input_ids, ins_kv_indices)
            self.logits_cache[req.rid] = LogitsCache(
                page_size=1, disable=False)
            self.consumer_queue[req.rid] = Queue()

    def summary(self, rid: str) -> Optional[Tuple[torch.Tensor, List[int]]]:
        assert rid in self.req_logits_key, "rid must stored at req_logits_key"
        logits_seq = self.req_logits_key[rid]
        history_logits = torch.stack([key.logits for key in logits_seq], dim=0)
        token_ids = [
            key.logits_result.item() if isinstance(
                key.logits_result, torch.Tensor) else key.logits_result
            for key in logits_seq
        ]
        return history_logits, token_ids

    def update_req(self, req: Req, history_logits: torch.Tensor):
        # in case of overlap mode
        assert len(req.output_ids) <= history_logits.shape[0]
        logits_token_ids = req.output_ids
        history_logits = history_logits[:len(req.output_ids)]
        cache_tree = self.logits_cache[req.rid]
        ind = self.select_topk_logits(history_logits, top_k=5)
        hit_cnt = cache_tree.insert(LogitsKey(
            history_logits, logits_token_ids), spotNodes=ind)
        cache_tree.init_time = time.monotonic()  # update timing
        # gen nxt logits
        self.producer_queue.put(req)
        # clear request info
        self.req_logits_key.pop(req.rid)
        if req.rid in self.req_already_loaded:
            self.req_already_loaded.remove(req.rid)
            
        # update req's logits cache info
        req.logits_cache_hit += hit_cnt
        req.logits_cache_budget += len(logits_token_ids)

    def produce_nxt_logits(self):
        """
        Once a request is done, we can pre-compute it's next logits result
        """
        while True:
            req = self.producer_queue.get()

            if req.rid not in self.logits_cache:
                continue
            if req.rid in self.req_already_loaded:
                continue

            e = self.event_pool.setdefault(req.rid, threading.Event())
            e.clear()
            lo_cache = self.logits_cache[req.rid]
            node = lo_cache.root_node
            sampling_batch_info = self.generate_sampling_info([req])
            _sample = partial(self.sample_child,
                              sampling_batch_info, req.seqlen)
            # this may consume much time
            output_tok, last_token_logits = lo_cache._resampling_normal(
                node, _sample, req)
            # output_tok, last_token_logits = lo_cache._resampling_spot_nodes(
            #     node, _sample, req)
            self.consumer_queue[req.rid].put((output_tok, last_token_logits))
            e.set()

    def get_logits_cache(self, req: Req) -> Tuple[bool, bool]:
        if req.rid not in self.req_info:
            return False, False
        self.event_pool[req.rid].wait()
        req_info = self.req_info[req.rid]
        con_q = self.consumer_queue[req.rid]
        eos_set = req.eos_token_ids
        try:
            output_tok, last_token_logits = con_q.get_nowait()
        except Empty:
            return False, False

        # Here we should first check if kv cache is still stored at radix tree
        if self.tree_cache is not None:
            req.fill_ids = req_info.input_tok.tolist() + output_tok
            input_len = len(req.fill_ids)
            max_prefix_len = input_len - 1
            max_prefix_len = max(max_prefix_len, 0)
            token_ids = req.fill_ids[:max_prefix_len]
            
            match_result = self.tree_cache.match_prefix(
                key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),
                **(
                    {"req": self, "cow_mamba": True}
                    if isinstance(self.tree_cache, MambaRadixCache)
                    else {}
                ),
            )
            
            self.log_cache_info(req, len(token_ids)-1)
            (
                req.prefix_indices,
                req.last_node,
                req.last_host_node,
                req.host_hit_length,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.host_hit_length,
            )

        # update request info
        req.output_ids = output_tok
        req.origin_input_ids = req_info.input_tok.tolist()
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.cache_protected_len = len(req.fill_ids) - 1
        req.last_token_logits = last_token_logits
        
        self.req_already_loaded.add(req.rid)
        # update request type
        req.r_type = 3  # Req_type.SAMPLING_DONE
        return True, output_tok[-1] in eos_set

    def log_cache_info(self, req: Req, cached_tok: int):
        msg = f"Request {req.rid}, #cached-token: {cached_tok}"
        logger.info(msg)

    def generate_sampling_info(self, reqs: List[Req]) -> SamplingBatchInfo:
        global_server_args = get_global_server_args()
        enable_deterministic = global_server_args.enable_deterministic_inference
        vocab_size = self.model_config.vocab_size

        temperatures = torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            device='cpu'
        ).view(-1, 1)
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float, device='cpu'
        )
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int32, device='cpu'
        )
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs], dtype=torch.float, device='cpu'
        )
        sampling_seed = (
            torch.tensor(
                [r.sampling_params.sampling_seed for r in reqs],
                dtype=torch.int32,
                device='cpu'
            )
            if enable_deterministic
            else None
        )

        logit_bias = None
        if any(r.sampling_params.logit_bias is not None for r in reqs):
            logit_bias = torch.zeros(len(reqs), vocab_size, device="cpu")
            for i, r in enumerate(reqs):
                if r.sampling_params.logit_bias is not None:
                    for key, value in r.sampling_params.logit_bias.items():
                        logit_bias[i, int(key)] = value

        merged_custom_logit_processor = None
        custom_params = None
        return SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            sampling_seed=sampling_seed,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_top_p_sampling=any(
                r.sampling_params.top_p != 1.0 for r in reqs),
            need_top_k_sampling=any(
                r.sampling_params.top_k != TOP_K_ALL for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            vocab_size=vocab_size,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device='cpu',
            logit_bias=logit_bias,
        )

    def select_topk_logits(
        self,
        logits: torch.Tensor,
        top_k: int,
    ) -> List:
        # Compute per-step entropy for the logits sequence.
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        # confidence
        p_max = probs.max(dim=-1).values
        # importance score
        importance = entropy * (1 - p_max)
        # normalize
        importance = importance.clamp(min=1e-8)
        # multiply with time step
        t = torch.arange(len(entropy), device=entropy.device)
        time_weight = 1 / (1 + 0.002 * t)
        importance *= time_weight

        weights = importance / importance.sum()
        indices = torch.topk(weights, top_k).indices
        return indices.tolist()


@dataclass
class LogitsKey:
    logits: Union[torch.Tensor, List]
    logits_result: Union[torch.Tensor, List]

    def __post_init__(self):
        if torch.is_tensor(self.logits) and self.logits.is_cuda:
            self.logits = self.logits.to("cpu", non_blocking=True)
        if torch.is_tensor(self.logits_result) and self.logits_result.is_cuda:
            self.logits_result = self.logits_result.to(
                "cpu", non_blocking=True)

    def __len__(self) -> int:
        return len(self.logits)

    def __iter__(self) -> Iterator[int]:
        return iter(self.logits)

    def __getitem__(self, idx: Union[int, slice]) -> LogitsKey:
        if isinstance(idx, int):
            seq_len = len(self)
            normalized_idx = idx if idx >= 0 else seq_len + idx
            if normalized_idx < 0 or normalized_idx >= seq_len:
                raise IndexError("LogitsKey index out of range")
            sliced = slice(normalized_idx, normalized_idx + 1)
            return LogitsKey(self.logits[sliced], self.logits_result[sliced])

        return LogitsKey(self.logits[idx], self.logits_result[idx])


class TreeNode:
    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: LogitsKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()
        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    @property
    def is_null(self) -> bool:
        return (self.key, self.value) == (None, None)

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match_page_size1(key0: LogitsKey, key1: LogitsKey):
    # _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.logits_result, key1.logits_result):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: LogitsKey, key1: LogitsKey, page_size: int):
    # _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.logits_result[i: i + page_size] != key1.logits_result[i: i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: LogitsKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.logits_result[0]
    else:
        plain_key = tuple(key.logits_result[:page_size])
    return plain_key


class LogitsCache(BasePrefixCache):
    def __init__(
        self,
        page_size: int,
        # sample_info: SamplingParams,
        disable: bool = False,
        enable_metrics: bool = False,
        enable_kv_cache_events: bool = False,
        eviction_policy: str = "lru",
        is_eagle: bool = False,
    ):
        self.page_size = page_size
        self.disable = disable
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []
        self.is_eagle = is_eagle
        # record spot index -> singleton treenode
        # each index map to spot tree node
        self._spot_index_to_node: Dict[int, TreeNode] = {}
        self._spot_node_to_offset: DefaultDict[TreeNode, list[int]] = defaultdict(
            list)  # each node map to spot logits offset

        if enable_metrics:
            self.init_metrics_collector()

        self.device = torch.device("cpu")
        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=page_size)

        if eviction_policy.lower() == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif eviction_policy.lower() == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif eviction_policy.lower() == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif eviction_policy.lower() == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif eviction_policy.lower() == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo'."
            )
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = None
        self.root_node.value = None
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.init_time = time.monotonic()  # used to evict request info
        self._record_all_cleared_event()

    def insert(self, key: LogitsKey, value: torch.Tensor = None, spotNodes: List = []):
        if self.disable:
            return 0
        if value is None:
            value = torch.tensor(key.logits_result, dtype=torch.int64)
        return self._insert_helper(self.root_node, key, value, spotNodes=spotNodes)

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int):
        if self.disable:
            return

        start_time = time.perf_counter()
        leaves = self._collect_leaves()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            if node.parent is None:
                assert (
                    node is self.root_node
                ), "This request holds the node from another tree"
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _resampling_normal(
        self,
        node: TreeNode,
        _sample: Callable[[Union[torch.Tensor, TreeNode]], int],
        req: Req,
    ) -> Tuple[List[int], Optional[torch.Tensor]]:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        root_logit = LogitsProcessorOutput(next_token_logits=node.key.logits)
        child_key = _sample(root_logit)
        child_key = child_key.item()
        output_tok: List[int] = [child_key]
        last_token_logits: Optional[torch.Tensor] = root_logit.next_token_logits
        eos_set = req.eos_token_ids
        should_exit = False

        while not should_exit:
            child = node.children.get(child_key)
            if child is None:
                break

            logits_res = child.key.logits_result
            for idx in range(1, len(logits_res)):
                cur_logit = LogitsProcessorOutput(
                    next_token_logits=child.key.logits[idx:idx + 1]
                )
                last_token_logits = cur_logit.next_token_logits
                nxt_id = _sample(cur_logit).item()
                output_tok.append(nxt_id)

                if logits_res[idx] != nxt_id:
                    should_exit = True
                    break

                if nxt_id in eos_set:
                    should_exit = True
                    break

            node = child
            if should_exit:
                break
            child_key = nxt_id

        return output_tok, last_token_logits

    def _resampling_spot_nodes(
        self,
        node: TreeNode,
        _sample: Callable[[Union[torch.Tensor, TreeNode]], int],
        req: Req,
    ) -> Tuple[List[int], Optional[torch.Tensor]]:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        output_tok: List[int] = []
        last_token_logits: Optional[torch.Tensor] = None
        eos_set = req.eos_token_ids
        should_exit = False

        def get_child_key(node: TreeNode) -> int:
            if node in self._spot_node_to_offset:
                ...
            else:
                assert len(
                    node.children) == 1, "Non-spot node should have only one child"
                return next(iter(node.children.keys()))

        child_key = get_child_key(node)
        while not should_exit:
            child = node.children.get(child_key)
            if child is None:
                break
            if child not in self._spot_node_to_offset:
                output_tok.extend(child.key.logits_result)
                if len(child.key.logits_result) > 0:
                    last_token_logits = child.key.logits[len(child.key.logits_result) - 1: len(child.key.logits_result)]
            else:
                logits_res = child.key.logits_result

                offset = self._spot_node_to_offset[child]
                base_start = 0
                for off in offset:
                    output_tok.extend(logits_res[base_start:off])

                    off_logit = LogitsProcessorOutput(
                        next_token_logits=child.key.logits[off:off + 1]
                    )
                    last_token_logits = off_logit.next_token_logits
                    nxt_id = _sample(off_logit).item()
                    if logits_res[off] != nxt_id:
                        output_tok.append(nxt_id)
                        should_exit = True
                        break

                    if nxt_id in eos_set:
                        output_tok.append(nxt_id)
                        should_exit = True
                        break

                    base_start = off
                node = child
                if should_exit:
                    break
                child_key = nxt_id

        return output_tok, last_token_logits

    def _split_node(self, key: LogitsKey, child: TreeNode, split_len: int):
        # new_node -> child
        if split_len == 0:
            return child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        return new_node

    @staticmethod
    def _advance_spot_ptr(split_spots: List[int], spot_ptr: int, consumed_len: int) -> int:
        while spot_ptr < len(split_spots) and split_spots[spot_ptr] <= consumed_len:
            spot_ptr += 1
        return spot_ptr

    def _insert_helper(self, node: TreeNode, key: LogitsKey, value: torch.Tensor, spotNodes: List):
        access_time = time.monotonic()
        node.last_access_time = access_time
        spotNodes.sort()
        if len(key) == 0:
            return 0

        # first init root node is null, we put first logits into root
        if node.is_null:
            node.key = key[0]
            node.lock_ref += 1
            self.evictable_size_ += 1

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0
        start_index = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key.logits_result):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._record_store_event(new_node)
            for ind in spotNodes:
                self._spot_index_to_node[ind] = new_node
                self._spot_node_to_offset[new_node].append(ind - start_index)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.logits_result[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        pass
        # for k, v in node.parent.children.items():
        #     if v == node:
        #         break
        # del node.parent.children[k]
        # self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = list(self.root_node.children.values())

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                if cur_node.lock_ref == 0:
                    ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _record_store_event(self, node: TreeNode):
        return

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start: start + self.page_size]
                if not page_tokens:
                    continue
                block_hash = hash(tuple(page_tokens))
                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash]))

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        pass

    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    def match_prefix(self, key: LogitsKey, **kwargs) -> MatchResult:
        pass


# if __name__ == "__main__":
#     tree = LogitsCache(page_size=1, disable=False)

#     # Example token id sequences (as lists of ints)
#     tree.insert("123", LogitsKey(logits=[1, 2, 3], logits_result=0))
#     tree.insert("123", LogitsKey(logits=[1, 2, 3], logits_result=1))
#     tree.insert("123", LogitsKey(logits=[1, 2, 4, 5], logits_result=1))
#     tree.insert("123", LogitsKey(logits=[1, 2, 4, 5, 6, 7], logits_result=1))
#     tree.insert("123", LogitsKey(logits=[8, 9, 10, 11, 12], logits_result=1))
#     tree.pretty_print()

#     # print(tree.match_prefix(LogitsKey(token_ids=[1, 2, 3, 13, 14], extra_key=None)))
