from __future__ import annotations
import asyncio
import threading
from collections import defaultdict
from typing import Any, Dict, NewType, Optional
from aiohttp import web
import requests
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
    logits_cached_toks: Optional[int] = None

    def __init__(self,
                 cached_toks: int,
                 input_toks: int,
                 output_toks: int,
                 logits_cached_toks: int = 0):
        self.cached_toks = cached_toks
        self.input_toks = input_toks
        self.output_toks = output_toks
        self.logits_cached_toks = logits_cached_toks

    def __add__(self, other: PriorityScore) -> PriorityScore:
        return PriorityScore(
            cached_toks=self.cached_toks + other.cached_toks,
            input_toks=self.input_toks + other.input_toks,
            output_toks=self.output_toks + other.output_toks,
            logits_cached_toks=self.logits_cached_toks + other.logits_cached_toks,
        )


class SglAgentPool:
    _lock = threading.Lock()
    _agents: Dict[AGENT_ID, Dict[str, Any]] = {}

    # record each agent instance's GPU budget
    agent_budget: Dict[AGENT_ID, Dict[GPU_ID, PAGE_NUM]] = {}
    # record each GPU's remain page size
    device_budget: Dict[GPU_ID, PAGE_NUM] = {}
    # record each agent instance score, which will be caculated to update importance score
    agent_score: Dict[AGENT_ID, PriorityScore] = defaultdict(
        lambda: PriorityScore(
            cached_toks=0,
            input_toks=0,
            output_toks=0,
            logits_cached_toks=0,
        )
    )

    def __init__(self, server_addr: str, gpu_id: int, req_to_token_pool: ReqToTokenPool):
        host, port = server_addr.split(":")
        self.register_server = SglAgentRegisterServer(
            host=host, port=int(port))
        self.req_to_token_pool = req_to_token_pool

        with self._lock:
            self.device_budget[gpu_id] = req_to_token_pool.size

    @classmethod
    def _rebalance_budget_locked(cls):
        agent_ids = list(cls._agents.keys())
        agent_cnt = len(agent_ids)
        if agent_cnt == 0:
            cls.agent_budget.clear()
            return

        cls.agent_budget = {AGENT_ID(agent_id): {} for agent_id in agent_ids}
        for gpu_id, total_mem in cls.device_budget.items():
            per_agent = int(total_mem) // agent_cnt
            remainder = int(total_mem) % agent_cnt
            for idx, agent_id in enumerate(agent_ids):
                extra = 1 if idx < remainder else 0
                cls.agent_budget[AGENT_ID(agent_id)][GPU_ID(
                    gpu_id)] = PAGE_NUM(per_agent + extra)

    @classmethod
    def _normalize_agent_metadata(
        cls,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        assert agent_id is not None, "Agent must have an ID"

        normalized_agent_id = str(agent_id)
        normalized_metadata = dict(metadata or {})
        return normalized_agent_id, normalized_metadata

    @classmethod
    def _serialize_budget_locked(cls, agent_id: Optional[str] = None):
        def _serialize(agent_key: AGENT_ID):
            budget = cls.agent_budget.get(agent_key, {})
            return {int(gpu_id): int(page_num) for gpu_id, page_num in budget.items()}

        if agent_id is not None:
            return _serialize(AGENT_ID(agent_id))

        return {
            str(agent_key): _serialize(agent_key)
            for agent_key in cls.agent_budget
        }

    @property
    def total_budget(self):
        return sum(self.device_budget.values())

    @classmethod
    def register(cls, agent: Any) -> str:
        agent_uuid, metadata = cls._normalize_agent_metadata(agent=agent)
        with cls._lock:
            cls._agents[AGENT_ID(agent_uuid)] = metadata
            # allocate budget online
            cls._rebalance_budget_locked()

        return agent_uuid

    @classmethod
    def register_agent(
        cls,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        normalized_agent_id, normalized_metadata = cls._normalize_agent_metadata(
            agent_id=agent_id,
            metadata=metadata,
        )
        with cls._lock:
            cls._agents[AGENT_ID(normalized_agent_id)] = normalized_metadata
            cls._rebalance_budget_locked()
            cls._serialize_budget_locked(normalized_agent_id)

    @classmethod
    def get(cls, agent_uuid: str):
        with cls._lock:
            return cls._agents.get(AGENT_ID(str(agent_uuid)))

    @classmethod
    def get_budget(cls, agent_uuid: str):
        with cls._lock:
            return cls._serialize_budget_locked(str(agent_uuid))

    def collect_agent_usage(self, req: Req, agent_uuid: Optional[str] = None):
        # TODO: add logits cache tokens
        if agent_uuid is None:
            return
        
        if agent_uuid not in self._agents:
            logger.warning(
                f"Agent {agent_uuid} not found in pool during usage collection")
            return
        
        cached_tokens = len(req.prefix_indices or [])
        input_tokens = len(req.fill_ids or [])
        output_tokens = len(req.output_ids or [])
        self.agent_score[AGENT_ID(agent_uuid)] += PriorityScore(
            cached_toks=cached_tokens,
            input_toks=input_tokens,
            output_toks=output_tokens,
            logits_cached_toks=0,
        )


class SglAgent:
    def __init__(self, agent):
        self.agent_cls = agent

    def __call__(self, *args, **kwargs):
        instance = self.agent_cls(*args, **kwargs)
        agent_id = getattr(instance, "id", None)
        assert agent_id is not None, "Agent must have an ID"

        rpc_endpoint = getattr(instance, "rpc_endpoint", None)
        if rpc_endpoint is None:
            raise RuntimeError(
                "Agent instance must have rpc_endpoint attribute for registration")

        self.register(rpc_endpoint, agent_id)
        return instance

    def register(self, server_addr: str, agent_id: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            response = requests.put(
                f"{server_addr}/register",
                json={"agent_id": str(agent_id), "metadata": metadata},
                timeout=5000,
            )
            if response.status_code == 200:
                logger.info(f"Successfully registered agent {agent_id}")
            else:
                logger.error(
                    f"Failed to register agent {agent_id}: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error registering agent {agent_id} due to: {e}")
            return None


class SglAgentRegisterServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.init_server()
        # Start register server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

    def init_server(self):
        self.app.router.add_route("*", "/register", self.register_agent)

    async def register_agent(self, request: web.Request):
        method = request.method
        if method == "PUT":
            data = await request.json()
            agent_id = data.get("agent_id")
            metadata = data.get("metadata")
            SglAgentPool.register_agent(agent_id, metadata)
            return web.Response(
				text=f"success",
				status=200,
				content_type="application/json",
			)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
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
