import os

import torch.distributed as dist
import torch


def run_sequential(rank, size, num_iter=10):
    """
    Prints the process rank sequentially in two orders over `num_iter` iterations,
    separating the output for each iteration by `---`.
    Example (3 processes, num_iter=2):
    ```
    Process 0
    Process 1
    Process 2
    Process 2
    Process 1
    Process 0
    ---
    Process 0
    Process 1
    Process 2
    Process 2
    Process 1
    Process 0
    ```
    """
    tensor = torch.empty((1,))

    if rank == 0:
        print('```')

    for _ in range(num_iter):
        if rank == 0:
            if num_iter != 0:
                print('---')
            print(f'Process {rank}')
            dist.send(tensor, 1)
            dist.recv(tensor, 1)
            print(f'Process {rank}')
        elif rank == size - 1: 
            dist.recv(tensor, rank - 1)
            print(f'Process {rank}')
            print(f'Process {rank}')
            dist.send(tensor, size - 2)
        else:
            dist.recv(tensor, rank - 1)
            print(f'Process {rank}')
            dist.send(tensor, rank + 1)
            dist.recv(tensor, rank + 1)
            print(f'Process {rank}')
            dist.send(tensor, rank - 1)

    if rank == 0:
        print('```')

    


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(rank=local_rank, backend="gloo")

    run_sequential(local_rank, dist.get_world_size())
