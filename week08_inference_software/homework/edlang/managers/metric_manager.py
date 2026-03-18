import time
import torch


METRIC_SHOW_PERIOD = 3.0

class MetricManager:
    def __init__(self, enable_metrics: bool = False):

        self.enable_metrics = enable_metrics
        self.waiting_queue_num = 0
        self.active_requests_num = 0

        self.throughput_tokens_per_second = 0.0
        self.ttft_ms = 0.0
        self.tpot_ms = 0.0
        self.rps = 0.0
        self.time = time.time()

        self.last_show_time = self.time
        self.total_generated_tokens = 0
        self.total_decode_time_s = 0.0
        self.total_completed_requests = 0
        self.total_latency_ms = 0.0
        self.total_queue_wait_ms = 0.0
        self.total_engine_time_s = 0.0
        self.total_wall_time_s = 0.0
        self.last_update_time = self.time
        self.gpu_util_approx = 0.0
        self.end_to_end_ms = 0.0
    
        self.request_start_times = {}
        self.first_token_times = {}

    def register_request_start(self, request_id: int):
        self.request_start_times[request_id] = time.time()

    def register_first_token(self, request_id: int):
        if request_id not in self.first_token_times:
            self.first_token_times[request_id] = time.time()

    def register_request_finish(self, request_id: int):
        self.total_completed_requests += 1

        t_start = self.request_start_times.get(request_id)
        if t_start is not None:
            latency_ms = (time.time() - t_start) * 1000.0
            self.total_latency_ms += latency_ms

    def register_decode_step(self, tokens_generated: int, decode_time_s: float):
        self.total_generated_tokens += tokens_generated
        self.total_decode_time_s += decode_time_s

    def register_queue_leave(self, request_id: int):
        t_start = self.request_start_times.get(request_id)
        if t_start is not None:
            wait_ms = (time.time() - t_start) * 1000.0
            self.total_queue_wait_ms += wait_ms

    def register_engine_step(self, step_time_s: float):
        self.total_engine_time_s += step_time_s
        now = time.time()
        self.total_wall_time_s += (now - self.last_update_time)
        self.last_update_time = now

    def calculate_throughtput_tokens_per_second(self, tokens_num: int, time_s: float):
        # TODO: Implement throughput calculation
        if time_s > 0:
            self.throughput_tokens_per_second = tokens_num / time_s
        else:
            self.throughput_tokens_per_second = 0.0

    def update_waiting_queue_num(self, num: int):
        # TODO: Implement waiting queue number update
        self.waiting_queue_num = num

    def update_active_requests_num(self, num: int):
        # TODO: Implement active requests number update
        self.active_requests_num = num

    def set_no_work(self):
        # TODO: Implement no work state update
        if not self.enable_metrics:
            return
        now = time.time()
        if now - self.last_show_time >= METRIC_SHOW_PERIOD:
            self.show_metrics(stage="IDLE")
            self.last_show_time = now

    def _update_latency_metrics(self):
        # TTFT
        ttfts = []
        for rid, t_first in self.first_token_times.items():
            t_start = self.request_start_times.get(rid)
            if t_start is not None:
                ttfts.append((t_first - t_start) * 1000.0)
        if ttfts:
            self.ttft_ms = sum(ttfts) / len(ttfts)

        # TPOT
        if self.total_generated_tokens > 0:
            self.tpot_ms = (self.total_decode_time_s / self.total_generated_tokens) * 1000.0
        else:
            self.tpot_ms = 0.0

        # RPS
        elapsed = time.time() - self.time
        if elapsed > 0:
            self.rps = self.total_completed_requests / elapsed
        else:
            self.rps = 0.0

        self.calculate_throughtput_tokens_per_second(
            self.total_generated_tokens,
            self.total_decode_time_s if self.total_decode_time_s > 0 else 1.0,
        )
        if self.total_completed_requests > 0:
            self.end_to_end_ms = self.total_latency_ms / self.total_completed_requests
        else:
            self.end_to_end_ms = 0.0

        if self.total_completed_requests > 0:
            self.avg_queue_wait_ms = self.total_queue_wait_ms / self.total_completed_requests
        else:
            self.avg_queue_wait_ms = 0.0

        if self.total_wall_time_s > 0:
            self.gpu_util_approx = self.total_engine_time_s / self.total_wall_time_s * 100.0
        else:
            self.gpu_util_approx = 0.0

    def show_metrics(self, stage: str):
        self._update_latency_metrics()
        gpu_mem_mb = 0.0
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        metrix_output = f"""
{stage}
- Throughput tokens per second: {self.throughput_tokens_per_second:.3f}
- TTFT: {self.ttft_ms:.3f} ms
- TPOT: {self.tpot_ms:.3f} ms
- RPS: {self.rps:.3f}
- End-to-end latency: {self.end_to_end_ms:.3f} ms
- Avg queue wait: {self.avg_queue_wait_ms:.3f} ms
- GPU memory (max): {gpu_mem_mb:.1f} MB
- GPU util (approx): {self.gpu_util_approx:.1f} %
- Waiting queue number: {self.waiting_queue_num}
- Active requests number: {self.active_requests_num}"""
        print("-" * 20 + metrix_output + "\n" + "-" * 20)
