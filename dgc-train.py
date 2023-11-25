import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torchvision

import pandas as pd
import time, datetime
from tqdm import tqdm
import utils

from dgcompress import DGC, DataPartitioner

#*******************************************************************************

approx_steps = 5000
workers = 4
batch_size = 64
learning_rate = 0.001

# Set the number of GPUs
world_size = 2
Method_Work = "ddp"  #  ddp / quantize / powersgd

Rank_Adder = 0

log_path = f"Logs/-dummy/exp-{Method_Work}-{world_size}gpu/"
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
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=64, shuffle=False, num_workers=workers)

    return train_loader, train_sampler, test_loader



def partition_dataset():
    """Load (or download) dataset and divide the data into partitions to feed into different branches
    train_set : function,  partition of the dataset depending on rank of the process
    b_size : int, batch size at each node
    """
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    ## Partition dataset to the number of parallel instances
    partition_size = [1. / world_size for _ in range(world_size)]
    train_partition = DataPartitioner(train_dataset, partition_size)
    train_partition = train_partition.use(dist.get_rank())
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size // world_size,
        shuffle=False,
        num_workers=workers,
        sampler=train_partition
    )

    test_dataset = torchvision.datasets.CIFAR10(
            root='../data',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        )
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=64, shuffle=False, num_workers=workers)

    return train_loader, train_partition, test_loader



def get_model_stuffs(rank):
    model = torchvision.models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

    model = model.to(rank)
    # dist_model = DistributedDataParallel(model, device_ids=[rank])
    dist_model = model
    return dist_model, model


def main(rank, world_size):
    utils.START_SEED()

    process_group = dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=120),
        world_size=world_size,
        rank=rank,
    )

    train_loader, train_sampler, test_loader = get_data_stuffs()
    dist_model, model = get_model_stuffs(rank)

    epochs = (approx_steps // len(train_loader))+1
    utils.LOG2TXT(f'{time.ctime()} epochs: {epochs}', file_path=log_path+"/logs.txt")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(dist_model.parameters(), lr=0.0001, momentum=0.9)

    dgc_trainer = DGC(model=model, rank=rank, size=world_size, device_id=rank,
                    momentum    = 0.9,  full_update_layers=[4],
                    percentages = [25, 6.25, 1.5625, 0.4, 0.1],
                    iterations  = [0, 50, 100, 200, 300])


    # Training loop
    st_time = time.time()
    step_count = 0
    loss_list = []
    mem_list  = []
    utz_list  = []
    for epoch in range(epochs):
        # train_sampler.set_epoch(epoch)
        for inputs, labels in tqdm(train_loader, disable= (rank!=0)):
            inputs, labels = inputs.to(rank), labels.to(rank)
            # Forward pass
            outputs = dist_model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            dgc_trainer.gradient_update(step_count)
            optimizer.step()

            step_count+=1
            loss_list.append(round(loss.item(), 6))
            mem_list.append(torch.cuda.memory_usage(rank+Rank_Adder))
            utz_list.append(torch.cuda.utilization(rank+Rank_Adder))

    ## Logging
    utils.LOG2TXT(f'GPU {rank}<-{rank+Rank_Adder} :: Time taken: {time.time()-st_time} \n'  +\
                f'Loss Start:{loss_list[0]} End:{loss_list[-1]}' +\
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

