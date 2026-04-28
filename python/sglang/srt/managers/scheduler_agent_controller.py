from __future__ import annotations
import asyncio
import json
import math
import threading
import multiprocessing as mp
import queue
import time
import uuid
from typing import Any, Dict, List, NewType, Optional
from aiohttp import web
import logging
from dataclasses import dataclass

import zmq

AGENT_ID = NewType("AGENT_ID", str)
GPU_ID = NewType("GPU_ID", int)
PAGE_NUM = NewType("PAGE_NUM", int)
logger = logging.getLogger(__name__)

AGENT_EVENT_ACK_TIMEOUT = 30.0
REMOTE_NODE_STARTUP_TIMEOUT = 30.0
REMOTE_AGENT_PORT_OFFSET = 1


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
        r"""
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
        node_rank: int = 0,
        nnodes: int = 1,
        dist_init_addr: Optional[str] = None,
    ):
        if not agent_server_addr:
            raise ValueError(
                "agent_server_addr must be set when agent serving is enabled"
            )
        self.host, self.port = agent_server_addr.split(":")
        self.port = int(self.port)
        self.app = web.Application()
        self.notify_queues = notify_queues or []
        self.ack_queue = ack_queue
        self.expected_receivers = expected_receivers
        self.node_rank = node_rank
        self.nnodes = nnodes
        self.dist_init_addr = dist_init_addr
        self._broadcast_lock = threading.Lock()
        self._remote_nodes: Dict[bytes, Dict[str, Any]] = {}
        self._remote_nodes_lock = threading.Lock()
        self._remote_command_queue: queue.Queue = queue.Queue()
        self._remote_bind_endpoint, self._remote_connect_endpoint = (
            self._build_remote_endpoints(agent_server_addr, dist_init_addr)
        )

        if self.node_rank == 0:
            self.init_server()
            if self.nnodes > 1:
                self.remote_thread = threading.Thread(
                    target=self._run_remote_router, daemon=True
                )
                self.remote_thread.start()
            # Start public register server only on head.
            self.thread = threading.Thread(target=self._run_server, daemon=True)
        else:
            # Non-head nodes keep a local bridge to their scheduler queues but do not
            # expose another public HTTP agent server.
            self.thread = threading.Thread(target=self._run_remote_worker, daemon=True)
        self.thread.start()

    def _build_remote_endpoints(
        self,
        agent_server_addr: str,
        dist_init_addr: Optional[str],
    ) -> tuple[str, str]:
        agent_host, agent_port = agent_server_addr.rsplit(":", 1)
        remote_port = int(agent_port) + REMOTE_AGENT_PORT_OFFSET

        connect_host = agent_host
        if dist_init_addr:
            connect_host = dist_init_addr.rsplit(":", 1)[0]
            if connect_host.startswith("[") and connect_host.endswith("]"):
                connect_host = connect_host[1:-1]
        elif agent_host in {"0.0.0.0", "::"}:
            connect_host = "127.0.0.1"

        return f"tcp://*:{remote_port}", f"tcp://{connect_host}:{remote_port}"

    def _broadcast_local_event(self, event: Dict[str, Any]) -> tuple[bool, List[Any]]:
        event_id = event["event_id"]

        for q in self.notify_queues:
            q.put(dict(event))

        acked: set[int] = set()
        deadline = time.monotonic() + AGENT_EVENT_ACK_TIMEOUT
        while len(acked) < len(self.expected_receivers):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                ack = self.ack_queue.get(timeout=remaining)
            except queue.Empty:
                break

            if ack.get("event_id") != event_id:
                continue
            if ack.get("status") == "ok":
                receiver_id = ack.get("receiver_id")
                acked.add(receiver_id)

        missing = [rid for rid in self.expected_receivers if rid not in acked]
        return len(missing) == 0, missing

    def _wait_for_remote_nodes(self) -> None:
        if self.nnodes <= 1:
            return

        deadline = time.monotonic() + REMOTE_NODE_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            with self._remote_nodes_lock:
                if len(self._remote_nodes) >= self.nnodes - 1:
                    return
            time.sleep(0.05)

    def _broadcast_remote_event(self, event: Dict[str, Any]) -> tuple[bool, List[Any]]:
        if self.nnodes <= 1:
            return True, []

        self._wait_for_remote_nodes()
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._remote_command_queue.put((dict(event), result_queue))
        try:
            return result_queue.get(timeout=AGENT_EVENT_ACK_TIMEOUT)
        except queue.Empty:
            with self._remote_nodes_lock:
                missing = [
                    rid
                    for node in self._remote_nodes.values()
                    for rid in node.get("receiver_ids", [])
                ]
            return False, missing

    def _broadcast_event(self, event: Dict[str, Any]) -> tuple[str, bool, List[Any]]:
        event_id = uuid.uuid4().hex
        event["event_id"] = event_id

        with self._broadcast_lock:
            local_ok, local_missing = self._broadcast_local_event(event)
            remote_ok, remote_missing = self._broadcast_remote_event(event)
            missing = local_missing + remote_missing
            return event_id, local_ok and remote_ok, missing

    def _handle_remote_message(self, identity: bytes, payload: Dict[str, Any]) -> None:
        op = payload.get("op")
        if op != "hello":
            return

        with self._remote_nodes_lock:
            self._remote_nodes[identity] = {
                "node_rank": payload.get("node_rank"),
                "receiver_ids": payload.get("receiver_ids", []),
            }

    def _recv_remote_payload(self, socket) -> tuple[bytes, Dict[str, Any]]:
        identity, raw_payload = socket.recv_multipart()
        return identity, json.loads(raw_payload.decode("utf-8"))

    def _run_remote_router(self):
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.bind(self._remote_bind_endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        try:
            while True:
                events = dict(poller.poll(50))
                if socket in events:
                    identity, payload = self._recv_remote_payload(socket)
                    self._handle_remote_message(identity, payload)

                try:
                    event, result_queue = self._remote_command_queue.get_nowait()
                except queue.Empty:
                    continue

                with self._remote_nodes_lock:
                    targets = dict(self._remote_nodes)

                for identity in targets:
                    socket.send_multipart(
                        [identity, json.dumps(event).encode("utf-8")]
                    )

                acked: set[bytes] = set()
                remote_missing: List[Any] = []
                missing_node_count = max((self.nnodes - 1) - len(targets), 0)
                remote_missing.extend(
                    f"node:{idx}:unconnected" for idx in range(missing_node_count)
                )
                deadline = time.monotonic() + AGENT_EVENT_ACK_TIMEOUT
                while len(acked) < len(targets):
                    remaining_ms = max(int((deadline - time.monotonic()) * 1000), 0)
                    if remaining_ms <= 0:
                        break
                    events = dict(poller.poll(remaining_ms))
                    if socket not in events:
                        continue
                    identity, payload = self._recv_remote_payload(socket)
                    if payload.get("op") == "hello":
                        self._handle_remote_message(identity, payload)
                        continue
                    if (
                        payload.get("op") != "ack"
                        or payload.get("event_id") != event["event_id"]
                    ):
                        continue
                    acked.add(identity)
                    remote_missing.extend(payload.get("missing_receivers", []))

                for identity, node in targets.items():
                    if identity not in acked:
                        remote_missing.extend(node.get("receiver_ids", []))

                result_queue.put((len(remote_missing) == 0, remote_missing))
        finally:
            socket.close(0)
            context.term()

    def _run_remote_worker(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt_string(
            zmq.IDENTITY, f"agent-node-{self.node_rank}-{uuid.uuid4().hex}"
        )
        socket.connect(self._remote_connect_endpoint)
        socket.send_json(
            {
                "op": "hello",
                "node_rank": self.node_rank,
                "receiver_ids": self.expected_receivers,
            }
        )

        try:
            while True:
                event = socket.recv_json()
                local_ok, missing = self._broadcast_local_event(event)
                socket.send_json(
                    {
                        "op": "ack",
                        "event_id": event.get("event_id"),
                        "status": "ok" if local_ok else "error",
                        "missing_receivers": missing,
                    }
                )
        finally:
            socket.close(0)
            context.term()

    def init_server(self):
        self.app.router.add_put("/register", self.register_agent)
        self.app.router.add_put("/unregister", self.unregister_agent)
        self.app.router.add_post("/rebalance", self.rebalance_agents)
        self.app.router.add_put("/rebalance", self.rebalance_agents)
        self.app.router.add_get("/heartbeat", self.heartbeat)
        self.app.router.add_get("/agent_info", self.get_agent_info)

    async def _maybe_read_json(self, request: web.Request) -> Dict[str, Any]:
        if request.can_read_body:
            try:
                data = await request.json()
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        return {}

    async def register_agent(self, request: web.Request):
        data = await self._maybe_read_json(request)
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
        data = await self._maybe_read_json(request)
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

    async def rebalance_agents(self, request: web.Request):
        data = await self._maybe_read_json(request)
        clear_scores = bool(data.get("clear_scores", False))
        event_id, ok, missing = self._broadcast_event(
            {"op": "rebalance", "clear_scores": clear_scores}
        )
        return web.json_response(
            {
                "status": "ok" if ok else "error",
                "event_id": event_id,
                "missing_receivers": missing,
                "clear_scores": clear_scores,
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
            if hasattr(self, "_runner"):
                self._loop.run_until_complete(self._runner.cleanup())
            if hasattr(self, "_loop"):
                self._loop.close()


def run_agent_register_server_process(
    agent_server_addr: str,
    notify_queues: List[mp.Queue],
    ack_queue: mp.Queue,
    expected_receivers: List[int],
    node_rank: int = 0,
    nnodes: int = 1,
    dist_init_addr: Optional[str] = None,
):
    server = SglAgentRegisterServer(
        agent_server_addr,
        notify_queues,
        ack_queue,
        expected_receivers,
        node_rank,
        nnodes,
        dist_init_addr,
    )
    server.thread.join()
