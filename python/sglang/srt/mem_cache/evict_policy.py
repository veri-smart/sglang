from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class AgentPriorityStrategy(EvictionStrategy):
    """Agent-aware eviction: evict lower-score and lower-reuse nodes first."""

    def __init__(self, get_agent_priority: Callable[[str], float]):
        self.get_agent_priority = get_agent_priority

    def get_priority(self, node: "TreeNode") -> Tuple[float, int, float]:
        agent_priority = 0.0
        if node.agent_ids:
            agent_priority = sum(
                self.get_agent_priority(agent_id) for agent_id in node.agent_ids
            )
        return (agent_priority, node.hit_count, node.last_access_time)
