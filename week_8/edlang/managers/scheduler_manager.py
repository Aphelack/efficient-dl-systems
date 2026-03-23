from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

import sys
import os

from edlang.entrypoints.engine import Request, InferenceEngine, BatchResult
from edlang.managers.metric_manager import MetricManager, METRIC_SHOW_PERIOD


@dataclass
class SchedulerConfig:
    max_batch_size: int = 8 
    max_waiting_requests: int = 100
    prefill_timeout_ms: float = 50.0
    enable_metrics: bool = False


class EDLangScheduler:

    def __init__(
        self,
        engine: InferenceEngine,
        config: Optional[SchedulerConfig] = None,
    ):
        self.engine = engine
        self.config = config or SchedulerConfig()

        self.waiting_queue = deque()
        self.active_requests = []

        self.next_request_id = 0
        self.metrics_manager = MetricManager(enable_metrics=self.config.enable_metrics)

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ):
        request = Request(
            request_id=self.next_request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        self.waiting_queue.append(request)

        self.metrics_manager.register_request_start(request.request_id)
        self.next_request_id += 1

        self.metrics_manager.update_waiting_queue_num(len(self.waiting_queue))
        
        return request.request_id
    
    def step(self):
        decode_result = None
        prefill_result = None
        step_start = time.time()
        # TODO: Implement step method
        # TODO: First decide how many requests to prefill
        
        prefill_result = self._prefill_step()

        if prefill_result is not None:
            for rid, finished in zip(prefill_result.request_ids, prefill_result.finished):
                self.metrics_manager.register_first_token(rid)
                if finished:
                    self.metrics_manager.register_request_finish(rid)
        # TODO: Then do decode
        active_before = [r for r in self.active_requests if not r.is_finished]
        start = time.time()
        decode_result = self._decode_step()
        end = time.time()
        

        if decode_result is not None:
            tokens_generated = 0
            for req in active_before:
                tokens_generated += 1
            self.metrics_manager.register_decode_step(tokens_generated, end - start)
            for rid, finished in zip(decode_result.request_ids, decode_result.finished):
                if finished:
                    self.metrics_manager.register_request_finish(rid)

        # TODO: Update metrics and inner state
        self.metrics_manager.update_waiting_queue_num(len(self.waiting_queue))
        
        self.metrics_manager.update_active_requests_num(
            len(self.active_requests)
        )
        # self.active_requests = [r for r in self.active_requests if not r.is_finished]
        step_end = time.time()
        self.metrics_manager.register_engine_step(step_end - step_start)
        if self.metrics_manager.enable_metrics:
            now = time.time()
            if now - self.metrics_manager.last_show_time >= METRIC_SHOW_PERIOD:
                self.metrics_manager.show_metrics(stage="STEP")
                self.metrics_manager.last_show_time = now

        # если нет работы
        if not self.waiting_queue and not any(not r.is_finished for r in self.active_requests):
            self.metrics_manager.set_no_work()
        self.metrics_manager._update_latency_metrics()

        return prefill_result, decode_result

        # TODO: Update metrics and inner state

    
    def _decode_step(self):        
        active = [req for req in self.active_requests if not req.is_finished]
        
        if not active:
            return None
        
        # TODO: Do decode for all active requests
        return self.engine.decode(active)
    
    def _prefill_step(self):
        if not self.waiting_queue:
            return None
        
        # Do prefill for some (which?) number of requests
        prefill_batch_size = self._decide_prefill_batch_size()
        for _ in range(prefill_batch_size):
            if not self.waiting_queue: break
            req = self.waiting_queue.popleft()
            self.active_requests.append(req)
            self.metrics_manager.register_queue_leave(req.request_id)
        return self.engine.prefill(self.active_requests)
    
    def _decide_prefill_batch_size(self):
        # The most simple policy: prefill only if there are no active requests
        num_active = len([r for r in self.active_requests if not r.is_finished])
        
        if num_active > 0:
            return 0
        else:
            return 1
    
    def get_finished_requests(self) -> List[Request]:
        finished = [req for req in self.active_requests if req.is_finished]
        self.active_requests = [req for req in self.active_requests if not req.is_finished]
        self.metrics_manager.update_active_requests_num(
            len(self.active_requests)
        )
        return finished

    
    def get_metric_manager(self):
        return self.metrics_manager

    def clear(self):
        self.waiting_queue = deque()
        self.active_requests = []
