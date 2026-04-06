from __future__ import annotations
import asyncio
import math
import threading
import multiprocessing as mp
import queue
import uuid
from typing import Any, Dict, List, NewType
from aiohttp import web
import logging
from dataclasses import dataclass
AGENT_ID = NewType("AGENT_ID", str)
GPU_ID = NewType("GPU_ID", int)
PAGE_NUM = NewType("PAGE_NUM", int)
logger = logging.getLogger(__name__)


@dataclass
class PriorityScore:
    cached_toks: int  # req kv cached toks
    input_toks: int   # req inputs toks (排除了 cached toks)
    output_toks: int  # req output toks
    agent_called_times: int
    logits_allocated_toks: int
    logits_hit_toks: int
    evict_times: int

    concurrent_reqs_sum: int
    concurrent_reqs_peak: int
    batch_presence: int

    def __init__(
        self,
        cached_toks: int = 0,
        input_toks: int = 0,
        output_toks: int = 0,
        agent_called_times: int = 0,
        logits_allocated_toks: int = 0,
        logits_hit_toks: int = 0,
        evict_times: int = 0,
        concurrent_reqs_sum: int = 0,
        concurrent_reqs_peak: int = 0,
        batch_presence: int = 0,
    ):
        self.cached_toks = cached_toks
        self.input_toks = input_toks
        self.output_toks = output_toks
        self.agent_called_times = agent_called_times
        self.logits_allocated_toks = logits_allocated_toks
        self.logits_hit_toks = logits_hit_toks
        self.evict_times = evict_times
        self.concurrent_reqs_sum = concurrent_reqs_sum
        self.concurrent_reqs_peak = concurrent_reqs_peak
        self.batch_presence = batch_presence

    def __add__(self, other: PriorityScore) -> PriorityScore:
        return PriorityScore(
            cached_toks=self.cached_toks + other.cached_toks,
            input_toks=self.input_toks + other.input_toks,
            output_toks=self.output_toks + other.output_toks,
            agent_called_times=self.agent_called_times + other.agent_called_times,
            logits_allocated_toks=self.logits_allocated_toks + other.logits_allocated_toks,
            logits_hit_toks=self.logits_hit_toks + other.logits_hit_toks,
            evict_times=self.evict_times + other.evict_times,
            concurrent_reqs_sum=self.concurrent_reqs_sum + other.concurrent_reqs_sum,
            concurrent_reqs_peak=max(
                self.concurrent_reqs_peak, other.concurrent_reqs_peak),
            batch_presence=self.batch_presence + other.batch_presence,
        )

    @property
    def logits_wasted_ratio(self) -> float:
        if self.logits_hit_toks == 0:
            return 0.0
        return self.logits_allocated_toks / self.logits_hit_toks

    @property
    def avg_concurrency(self) -> float:
        if self.batch_presence == 0:
            return 0.0
        return self.concurrent_reqs_sum / self.batch_presence

    @property
    def score(self) -> float:
        # Higher score => higher scheduling priority / larger target budget.
        cached_toks_weight = 5.0
        input_toks_weight = 3.0
        output_toks_weight = 2.0
        called_times_weight = 1.0
        logits_weight = 4.0
        evict_times_weight = 8.0

        _score = (
            self.cached_toks * cached_toks_weight
            + self.input_toks * input_toks_weight
            + self.output_toks * output_toks_weight
            + self.agent_called_times * called_times_weight
            - self.logits_wasted_ratio * logits_weight
            + self.evict_times * evict_times_weight
        )
        return _score

    @property
    def nonlinear_score(self) -> float:
        """
        U_i = activity * workload * efficiency * concurrency * eviction penalty
        ===>
        U_i = \log(1+\text{calls}_i)
        \cdot
        \sqrt{1+\text{input}_i+\text{output}_i}
        \cdot
        (1+\text{cache\_eff}_i+1.2\cdot \text{logits\_eff}_i)
        \cdot
        (1+0.5\log(1+\text{avg\_conc}_i)+0.3\log(1+\text{peak\_conc}_i))
        \cdot
        \frac{1}{1+0.5\cdot \text{evict\_pressure}_i}
        """
        input_toks = self.input_toks
        output_toks = self.output_toks
        cached_toks = self.cached_toks
        called_times = self.agent_called_times
        logits_allocated_toks = self.logits_allocated_toks
        logits_hit_toks = self.logits_hit_toks
        evict_times = self.evict_times
        concurrent_reqs_peak = self.concurrent_reqs_peak

        volume = input_toks + output_toks
        workload_term = math.sqrt(1.0 + volume)
        activity_term = math.log1p(called_times)

        cache_eff = cached_toks / (1.0 + input_toks + cached_toks)
        logits_eff = logits_hit_toks / (1.0 + logits_allocated_toks)

        concurrency_term = (
            1.0
            + 0.5 * math.log1p(self.avg_concurrency)
            + 0.3 * math.log1p(concurrent_reqs_peak)
        )

        evict_pressure = evict_times / (1.0 + called_times)
        eviction_term = 1.0 / (1.0 + 0.5 * evict_pressure)

        raw_score = (
            activity_term
            * workload_term
            * (1.0 + 1.0 * cache_eff + 1.2 * logits_eff)
            * concurrency_term
            * eviction_term
        )
        return raw_score


class SglAgentRegisterServer:
    def __init__(
        self,
        agent_server_addr: str,
        notify_queues: List[mp.Queue],
        ack_queue: mp.Queue,
        expected_receivers: List[int],
    ):
        self.host, self.port = agent_server_addr.split(":")
        self.port = int(self.port)
        self.app = web.Application()
        self.notify_queues = notify_queues or []
        self.ack_queue = ack_queue
        self.expected_receivers = expected_receivers
        self._broadcast_lock = threading.Lock()
        self.init_server()
        # Start register server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

    def _broadcast_event(self, event: Dict[str, Any]) -> tuple[str, bool, List[int]]:
        event_id = uuid.uuid4().hex
        event["event_id"] = event_id

        with self._broadcast_lock:
            for q in self.notify_queues:
                q.put(event)

            acked: set[int] = set()
            while len(acked) < len(self.expected_receivers):
                try:
                    ack = self.ack_queue.get(timeout=30000)
                except queue.Empty:
                    break

                if ack.get("event_id") != event_id:
                    continue
                if ack.get("status") == "ok":
                    receiver_id = ack.get("receiver_id")
                    acked.add(receiver_id)

            missing = [
                rid for rid in self.expected_receivers if rid not in acked]
            return event_id, len(missing) == 0, missing

    def init_server(self):
        self.app.router.add_put("/register", self.register_agent)
        self.app.router.add_put("/unregister", self.unregister_agent)
        self.app.router.add_get("/heartbeat", self.heartbeat)
        self.app.router.add_get("/agent_info", self.get_agent_info)

    async def register_agent(self, request: web.Request):
        data = await request.json()
        agent_id = data.get("agent_id")
        metadata = data.get("metadata")
        event_id, ok, missing = self._broadcast_event(
            {"op": "register", "agent_id": agent_id, "metadata": metadata}
        )
        return web.json_response(
            {
                "status": "ok" if ok else "error",
                "event_id": event_id,
                "missing_receivers": missing,
            },
            status=200 if ok else 504,
        )

    async def unregister_agent(self, request: web.Request):
        data = await request.json()
        agent_id = data.get("agent_id")
        event_id, ok, missing = self._broadcast_event(
            {"op": "unregister", "agent_id": agent_id}
        )
        return web.json_response(
            {
                "status": "ok" if ok else "error",
                "event_id": event_id,
                "missing_receivers": missing,
            },
            status=200 if ok else 504,
        )

    async def heartbeat(self, request: web.Request):
        return web.json_response(
            {
                "status": "ok",
                "host": self.host,
                "port": self.port,
                "slave_num": len(self.notify_queues),
            }
        )

    async def get_agent_info(self, request: web.Request):
        event_id, ok, missing = self._broadcast_event({"op": "agent_info"})
        return web.json_response(
            {
                "status": "ok" if ok else "error",
                "event_id": event_id,
                "missing_receivers": missing,
            },
            status=200 if ok else 504,
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


def run_agent_register_server_process(
    agent_server_addr: str,
    notify_queues: List[mp.Queue],
    ack_queue: mp.Queue,
    expected_receivers: List[int],
):
    server = SglAgentRegisterServer(
        agent_server_addr,
        notify_queues,
        ack_queue,
        expected_receivers,
    )
    server.thread.join()
