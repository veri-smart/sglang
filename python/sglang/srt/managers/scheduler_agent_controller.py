from __future__ import annotations
import asyncio
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
    cached_toks: int
    input_toks: int
    output_toks: int
    agent_called_times: int
    logits_cached_toks: int
    evict_times: int

    def __init__(
        self,
        cached_toks: int = 0,
        input_toks: int = 0,
        output_toks: int = 0,
        agent_called_times: int = 0,
        logits_cached_toks: int = 0,
        evict_times: int = 0,
    ):
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
        cached_toks_weight = 5
        input_toks_weight = 3
        output_toks_weight = 2
        logits_cached_toks_weight = 2
        return (
            self.cached_toks * cached_toks_weight
            + self.input_toks * input_toks_weight
            + self.output_toks * output_toks_weight
            + self.logits_cached_toks * logits_cached_toks_weight
            + self.evict_times
        )


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
                    ack = self.ack_queue.get(timeout=30)
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
