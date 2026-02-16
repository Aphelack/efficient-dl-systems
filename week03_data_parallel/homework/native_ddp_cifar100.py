import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


BATCH_SIZE = 64
EPOCHS = 10
WARMUP_STEPS = 20
ACCUM_STEPS = 2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        self.bn1 = nn.BatchNorm1d(128, affine=False)
        self.peak_memory = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)

    def update_peak_memory(self):
        current_memory = torch.cuda.memory_allocated()
        self.peak_memory = max(self.peak_memory, current_memory)

    def reset_peak_memory(self):
        self.peak_memory = 0
        torch.cuda.reset_peak_memory_stats()

    def get_peak_memory_mb(self):
        peak_stats = torch.cuda.max_memory_allocated()
        return max(self.peak_memory, peak_stats) / (1024 * 1024)


def compute_accuracy(model, dataset, batch_size, device, rank, world_size):
    model.eval()
    correct = 0
    total = 0

    sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    stats = torch.tensor([correct, total], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    return 100.0 * stats[0].item() / stats[1].item()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train_native(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = Net().to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    original_model = model.module

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    if rank == 0:
        CIFAR100("./cifar", train=True, transform=transform, download=True)
    dist.barrier()

    dataset = CIFAR100("./cifar", train=True, transform=transform, download=False)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_sampler = DistributedSampler(train_subset, world_size, rank, shuffle=True)
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    model.train()
    warmup_iter = iter(train_loader)
    for _ in range(WARMUP_STEPS):
        data, target = next(warmup_iter)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target) / ACCUM_STEPS
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    if rank == 0:
        torch.cuda.reset_peak_memory_stats()
    original_model.reset_peak_memory()

    start_time = time.time()

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad()

        iterator = tqdm(train_loader, disable=rank != 0)

        for step, (data, target) in enumerate(iterator):
            data, target = data.to(device), target.to(device)

            loss = criterion(model(data), target) / ACCUM_STEPS
            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            original_model.update_peak_memory()

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_batch_time = total_time / (EPOCHS * len(train_loader))
    memory_mb = original_model.get_peak_memory_mb()

    final_accuracy = compute_accuracy(
        model,
        val_subset,
        BATCH_SIZE,
        device,
        rank,
        world_size
    )

    stats = torch.tensor(
        [avg_batch_time, memory_mb, final_accuracy],
        device=device,
        dtype=torch.float64
    )

    dist.reduce(stats, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:

        report = {
            "benchmark_name": "native_syncbn_ddp",
            "num_gpus": world_size,
            "batch_size_per_gpu": BATCH_SIZE,
            "total_batch_size": BATCH_SIZE * world_size * ACCUM_STEPS,
            "avg_batch_time_seconds": stats[0].item(),
            "peak_memory_mb_per_gpu": stats[1].item(),
            "final_accuracy_percent": stats[2].item(),
        }

        with open(f"native_bn_report_w{world_size}.json", "w") as f:
            json.dump(report, f, indent=4)

        print(json.dumps(report, indent=4))

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_native, args=(world_size,), nprocs=world_size)
