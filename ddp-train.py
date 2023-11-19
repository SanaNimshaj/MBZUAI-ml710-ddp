import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision

from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
from torch.distributed.algorithms.ddp_comm_hooks import quantization_hooks
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (   size_based_auto_wrap_policy,
                                            enable_wrap,
                                            wrap,
                                        )

import pandas as pd
import time, datetime
from tqdm import tqdm
import utils

#*******************************************************************************

approx_steps = 20000
workers = 4
batch_size = 64
learning_rate =  0.0001 #0.0001

# Set the number of GPUs
world_size = 4
Method_Work = "quantize"  #  ddp / quantize / powersgd / fsdp / topk

Rank_Adder = 0

log_path = f"Logs/exp04-{Method_Work}-{world_size}gpu-32im-Res18/"
os.makedirs(log_path, exist_ok=True)

os.environ['MASTER_ADDR'] = ADDR = 'localhost'
os.environ['MASTER_PORT'] = PORT ='12345'

#*******************************************************************************

def get_data_stuffs():

    train_dataset = torchvision.datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size // world_size,
        shuffle=False,
        num_workers=workers,
        sampler=train_sampler
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=64, shuffle=False, num_workers=workers)

    return train_loader, train_sampler, test_loader


def get_model_stuffs(rank):
    model = torchvision.models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    model = model.to(rank)

    if Method_Work == "fsdp":
        my_auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=100000 )
        dist_model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)
    else:
        dist_model = DDP(model, device_ids=[rank])

    return dist_model, model

#*******************************************************************************


class TopKSparsificationState:
    def __init__(self, process_group, k_ratio):
        self.process_group = process_group
        self.k_ratio = k_ratio

    def sparsify_gradients(self, tensor):
        flat_tensor = tensor#.flatten()
        k_count = int( len(flat_tensor) // self.k_ratio )
        abs_tensor = flat_tensor.abs()
        topk_values, topk_indices = torch.topk(abs_tensor, k_count, sorted=False)
        mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        sparsified_tensor = flat_tensor * mask
        # sparsified_tensor = sparsified_tensor.reshape(tensor.shape)
        return sparsified_tensor.to_sparse()

def topk_sparsification_hook(state, bucket):
    process_group = state.process_group
    tensor = bucket.buffer()
    sparsified_tensor = state.sparsify_gradients(tensor)
    dist.all_reduce(sparsified_tensor, op=dist.ReduceOp.AVG,
                    group=process_group)
    sparsified_tensor.to_dense()

    return sparsified_tensor.get_future()
    # return default_hooks._allreduce_fut(process_group, sparsified_tensor)


class GradientQuantizationState:
    """TODO: Errorful state to be fixed """
    def __init__(self, process_group, quant_bits):
        self.process_group = process_group
        self.quant_bits = quant_bits

    def quantize(self, tensor):
        max_val = tensor.abs().max()
        scale = max_val / (2**self.quant_bits - 1)
        zero_point = 0
        quant_tensor = torch.round(tensor / scale) + zero_point
        quant_tensor = torch.clamp(quant_tensor, 0, 255).to(torch.uint8)
        return quant_tensor, scale

    def dequantize(quant_tensor, scale):
        zero_point = 0
        tensor = scale * (quant_tensor.to(torch.float32) - zero_point)
        return tensor

def gradient_quantization_hook(state, bucket):
    """TODO: Errorful state to be fixed """

    process_group = state.process_group
    tensor = bucket.buffer()
    quant_tensor, scale = state.quantize(tensor)
    dist.all_reduce(quant_tensor, op=dist.ReduceOp.AVG,
                    group=process_group)
    dist.all_reduce(scale, op=dist.ReduceOp.AVG,
                    group=process_group)
    dequant_tensor = state.dequantize(quant_tensor, scale)


    return dequant_tensor.get_future()


##==============================================================================


def main(rank, world_size):
    utils.START_SEED()

    process_group = dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=120),
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(rank)

    train_loader, train_sampler, test_loader = get_data_stuffs()
    dist_model, model = get_model_stuffs(rank)

    epochs = (approx_steps // len(train_loader))+1
    utils.LOG2TXT(f'{time.ctime()} epochs: {epochs}', file_path=log_path+"/logs.txt")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(dist_model.parameters(), lr=learning_rate, momentum=0.9)


    if Method_Work == "powersgd":
        powersgd_hook  = powerSGD_hook.powerSGD_hook
        powersgd_state = powerSGD_hook.PowerSGDState(process_group=process_group,
                 matrix_approximation_rank=4, start_powerSGD_iter=10,
                 min_compression_rate=0.5)

        dist_model.register_comm_hook(powersgd_state, powersgd_hook)

    elif Method_Work == "topk":
        topk_state = TopKSparsificationState(process_group, 10)
        dist_model.register_comm_hook(topk_state, topk_sparsification_hook)

    elif Method_Work == "quantize":
        # quant_state = GradientQuantizationState(process_group, 16)
        # dist_model.register_comm_hook(quant_state, gradient_quantization_hook)
        dist_model.register_comm_hook(process_group, quantization_hooks.quantization_pertensor_hook) ##-->>>



    # Training loop
    st_time = time.time()
    step_count = 0
    loss_list = []
    mem_list  = []
    utz_list  = []
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for inputs, labels in tqdm(train_loader, disable= (rank!=0)):
            inputs, labels = inputs.to(rank), labels.to(rank)
            # Forward pass
            outputs = dist_model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count+=1
            loss_list.append(round(loss.item(), 6))
            mem_list.append(torch.cuda.memory_usage(rank+Rank_Adder))
            utz_list.append(torch.cuda.utilization(rank+Rank_Adder))

    ## Logging
    utils.LOG2TXT(f'GPU {rank}<-{rank+Rank_Adder} :: Time taken: {time.time()-st_time} \n'  +\
                f'Loss Start:{loss_list[0]} End:{loss_list[-1]} \n' +\
                f'Steps taken: {step_count}'
                ,
                file_path=log_path+"/logs.txt")

    df = pd.DataFrame()
    df["loss"] = pd.Series(loss_list)
    df["mem"] = pd.Series(mem_list)
    df["utz"] = pd.Series(utz_list)
    df.to_csv(log_path+f"/step-details-{rank}.csv", index=False)

    dist.barrier()
    if rank == 0:
        ## TEST Loop
        dist_state = dist_model.state_dict()
        dist_state = { k.replace("module.", ""):v  for k,v in dist_state.items() }
        model.load_state_dict(dist_state)
        model = model.to(rank)
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / total_samples
        utils.LOG2TXT(f'Test Accuracy: {accuracy * 100:.2f}%', file_path=log_path+"/logs.txt")

    # Clean up
    dist.destroy_process_group()

if __name__ == '__main__':

    # Use the spawn method for multiprocessing with DDP
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

