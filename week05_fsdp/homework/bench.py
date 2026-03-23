import random
import socket
from collections import deque
from collections.abc import Callable
from typing import Any
from typing import Literal
import pathlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train import train

_PORT_MIN = 29500
_PORT_MAX = 30000


def _initialize_ports() -> deque[int]:
    ports = list(range(_PORT_MIN, _PORT_MAX))
    random.Random().shuffle(ports)
    return deque(ports)


_PORTS = _initialize_ports()


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _get_next_port() -> int:
    port = _PORTS[0]
    _PORTS.rotate()
    return port


def _find_open_port(host: str = "127.0.0.1", num_attempts: int = 10) -> int:
    port = _get_next_port()
    attempts = 0
    while _port_in_use(host, port):
        port += _get_next_port()
        attempts += 1
        if attempts >= num_attempts:
            raise RuntimeError("failed to find an open port")
    return port


def _run_test(
    rank: int,
    world_size: int,
    func: Callable,  # type: ignore[type-arg]
    func_args: tuple[Any],
    func_kwargs: dict[str, Any],
    primary_addr: str = "127.0.0.1",
    primary_port: int = 29500,
) -> None:
    torch.cuda.set_device(torch.device(f"cuda:{rank}"))
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{primary_addr}:{primary_port}",
        rank=rank,
        world_size=world_size,
        device_id=torch.cuda.current_device(),
    )
    try:
        func(*func_args, **func_kwargs)
    finally:
        dist.destroy_process_group()

def run_distributed_test[**P](
    func: Callable[P, None],
    world_size: int = 4,
    primary_addr: str = "127.0.0.1",
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    primary_port = _find_open_port()
    mp.start_processes(  # type: ignore[attr-defined,no-untyped-call]
        _run_test,
        args=(
            world_size,
            func,
            args,
            kwargs,
            primary_addr,
            primary_port,
        ),
        nprocs=world_size,
    )


def _test_fsdp(
    param_dtype: Literal["bfloat16", "float32"] | None,
    reduce_dtype: Literal["bfloat16", "float32"] | None,
    fsdp: Literal["fsdp2", "effdl"] | None = None
) -> None:
    train(
        num_steps_to_profile=3,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        fsdp=fsdp,
        snapshots_dir=pathlib.Path(f"snapshots_{fsdp}"),
        traces_dir=pathlib.Path(f"traces_{fsdp}")
    )


if __name__ == "main":
    run_distributed_test(
        _test_fsdp,
        param_dtype="bfloat16",
        reduce_dtype="bfloat16",
        fsdp='effdl',
        world_size=2,
    )