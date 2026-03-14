from __future__ import annotations
import asyncio
import threading
from collections import defaultdict
from typing import Any, Dict, NewType, Optional
from aiohttp import web
from dataclasses import dataclass
import logging
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.managers.schedule_batch import Req

AGENT_ID = NewType("AGENT_ID", str)
GPU_ID = NewType("GPU_ID", int)
PAGE_NUM = NewType("PAGE_NUM", int)
logger = logging.getLogger(__name__)


@dataclass
class PriorityScore:
    cached_toks: int
    input_toks: int
    output_toks: int
    agent_called_times: int
    logits_cached_toks: int
    evict_times: int

    def __init__(self,
                 cached_toks: int = 0,
                 input_toks: int = 0,
                 output_toks: int = 0,
                 agent_called_times: int = 0,
                 logits_cached_toks: int = 0,
                 evict_times: int = 0):
        self.cached_toks = cached_toks
        self.input_toks = input_toks
        self.output_toks = output_toks
        self.agent_called_times = agent_called_times
        self.logits_cached_toks = logits_cached_toks
        self.evict_times = evict_times

    def __add__(self, other: PriorityScore) -> PriorityScore:
        return PriorityScore(
            cached_toks=self.cached_toks + other.cached_toks,
            input_toks=self.input_toks + other.input_toks,
            output_toks=self.output_toks + other.output_toks,
            agent_called_times=self.agent_called_times + other.agent_called_times,
            logits_cached_toks=self.logits_cached_toks + other.logits_cached_toks,
            evict_times=self.evict_times + other.evict_times,
        )

    @property
    def score(self) -> int:
        # here we set stable weights for different token types to calculate the priority score
        cached_toks_weight = 5
        input_toks_weight = 3
        output_toks_weight = 2
        logits_cached_toks_weight = 2
        return (
            self.cached_toks * cached_toks_weight +
            self.input_toks * input_toks_weight +
            self.output_toks * output_toks_weight +
            self.logits_cached_toks * logits_cached_toks_weight +
            self.evict_times
        )


class SglAgentPool:
    _lock = threading.Lock()
    _agents: Dict[AGENT_ID, Dict[str, Any]] = {}

    # record each agent instance's page budget
    agent_budget: Dict[AGENT_ID, PAGE_NUM] = {}
    # record each agent instance score, which will be caculated to update importance score
    agent_score: Dict[AGENT_ID, PriorityScore] = defaultdict(PriorityScore)

    batch_size: int = 0
    FLUSH_THRESHOLD = 2

    def __init__(
        self,
        server_addr: str,
        req_to_token_pool: ReqToTokenPool,
        start_register_server: bool = True,
    ):
        host, port = server_addr.split(":")
        self.register_server = None
        if start_register_server:
            self.register_server = SglAgentRegisterServer(
                host=host, port=int(port), agent_pool=self
            )
        self.req_to_token_pool = req_to_token_pool
        self._total_budget = req_to_token_pool.size
        self.init_default()

    def init_default(self):
        if len(self._agents) > 0:
            return
        self.register_agent("default")

    def _rebalance_budget_locked(self):
        agent_ids = list(self._agents.keys())
        agent_cnt = len(agent_ids)
        if agent_cnt == 0:
            self.agent_budget.clear()
            return

        raw_scores = {
            agent_id: self.agent_score[agent_id].score
            for agent_id in agent_ids
        }
        total_score = sum(raw_scores.values())

        if total_score > 0:
            weights = {
                agent_id: raw_scores[agent_id] / total_score
                for agent_id in agent_ids
            }
        else:
            weights = {
                agent_id: 1.0 / agent_cnt
                for agent_id in agent_ids
            }

        alloced = {
            agent_id: int(self._total_budget * weights[agent_id])
            for agent_id in agent_ids
        }
        exact_alloc = {
            agent_id: self._total_budget * weights[agent_id]
            for agent_id in agent_ids
        }

        remainder = self._total_budget - sum(alloced.values())
        if remainder > 0:
            ordered = sorted(
                agent_ids,
                key=lambda aid: (
                    exact_alloc[aid] - alloced[aid],
                    str(aid),
                ),
                reverse=True,
            )
            for i in range(remainder):
                chosen = ordered[i % agent_cnt]
                alloced[chosen] += 1

        self.agent_budget = {
            agent_id: PAGE_NUM(alloced[agent_id])
            for agent_id in agent_ids
        }

    def _normalize_agent_metadata(
        self,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        assert agent_id is not None, "Agent must have an ID"

        normalized_agent_id = str(agent_id)
        normalized_metadata = metadata if metadata else {}
        return normalized_agent_id, normalized_metadata

    def register_agent(
        self,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        normalized_agent_id, normalized_metadata = self._normalize_agent_metadata(
            agent_id=agent_id,
            metadata=metadata,
        )
        with self._lock:
            self._agents[AGENT_ID(normalized_agent_id)] = normalized_metadata
            self._rebalance_budget_locked()
        logger.info(
            f"Registered agent {normalized_agent_id} with metadata {normalized_metadata}")

    def unregister_agent(self, agent_id: str):
        normalized_agent_id = str(agent_id)
        with self._lock:
            self._agents.pop(AGENT_ID(normalized_agent_id), None)
            self.agent_score.pop(AGENT_ID(normalized_agent_id), None)
            self._rebalance_budget_locked()
        logger.info(f"Unregistered agent {normalized_agent_id}")

    def get(self, agent_uuid: str):
        with self._lock:
            return self._agents.get(AGENT_ID(str(agent_uuid)))

    def collect_agent_usage(self, req: Req, agent_uuid: Optional[str] = None):
        # TODO: add logits cache tokens
        if agent_uuid is None:
            return

        if agent_uuid not in self._agents:
            logger.warning(
                f"Agent {agent_uuid} not found in pool during usage collection")
            return

        self.agent_score[AGENT_ID(agent_uuid)] += PriorityScore(
            cached_toks=len(req.prefix_indices),
            input_toks=len(req.fill_ids)-len(req.prefix_indices),
            output_toks=len(req.output_ids),
            agent_called_times=1,
            logits_cached_toks=0,
        )
        self.batch_size += 1
        if self.batch_size >= self.FLUSH_THRESHOLD:
            with self._lock:
                self._rebalance_budget_locked()
                self.agent_score.clear()
                self.batch_size = 0

    def remain_budget(self, req: Req) -> int:
        agent_id: str = req.agent_id if req.agent_id is not None else "default"
        if agent_id not in self.agent_budget:
            raise RuntimeError(f"Agent {agent_id} not found in budget")
        return self.agent_budget[agent_id]

class SglAgentRegisterServer:
    def __init__(self, host: str, port: int, agent_pool: SglAgentPool):
        self.host = host
        self.port = port
        self.agent_pool = agent_pool
        self.app = web.Application()
        self.init_server()
        # Start register server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

    def init_server(self):
        self.app.router.add_put("/register", self.register_agent)
        self.app.router.add_put("/unregister", self.unregister_agent)

    async def register_agent(self, request: web.Request):
        data = await request.json()
        agent_id = data.get("agent_id")
        metadata = data.get("metadata")
        self.agent_pool.register_agent(agent_id, metadata)
        return web.Response(
            text=f"success",
            status=200,
            content_type="application/json",
        )

    async def unregister_agent(self, request: web.Request):
        data = await request.json()
        agent_id = data.get("agent_id")
        self.agent_pool.unregister_agent(agent_id)
        return web.Response(
            text=f"success",
            status=200,
            content_type="application/json",
        )

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            access_log = None
            if logger.getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, host=self.host, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()
