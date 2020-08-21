#https://pytorch.org/docs/master/notes/ddp.html
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyData(Dataset):
    def __init__(self, num_sample=200):
        self.num_sample = num_sample
        self.data_list = []
        self.label_list = []
        for i in range(self.num_sample):
            self.data_list.append(torch.randn(10))
            self.label_list.append(torch.randn(10))
            


    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        return {'data': self.data_list[idx], 'label': self.label_list[idx]}


def example(rank, world_size):
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    print(model.weight[0,0])
    #torch.cuda.set_device(rank)
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    print('Rank: %d, ddp model: %s'%(rank, str(ddp_model.module.weight[0,0])))
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    dataset = ToyData(1000)
    data_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank
            )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=data_sampler)
    if rank == 0:
        torch.save(ddp_model.state_dict(), 'model.pth')
    for epoch in range(1):
        for i,batch in enumerate(dataloader):
            # forward pass
            ## set non_blocking to be False, otherwise, the programe will hanging
            data = batch['data'].to(rank)
            label = batch['label'].to(rank)
            outputs = ddp_model(data)
            #outputs = ddp_model(torch.randn([10, 10]).to(rank, non_blocking=True))
            #label = torch.randn([10, 10]).to(rank, non_blocking=True)
            # backward pass
            loss = loss_fn(outputs, label)
            loss.backward()
            # update parameters
            optimizer.step()
            if i == 100:
                if rank <= 100:
                    ckpt = torch.load('model.pth', map_location='cpu')
                    ddp_model.load_state_dict(ckpt)
                print('In iteration, Rank: %d, ddp model: %s'%(rank, str(ddp_model.module.weight[0,0])))
        if rank == 0:
            print('batch: %d, loss: %f'%(i, loss.item()))
def main():
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='8888'

########################
    world_size = 4

    mp.spawn(example,
             args=(world_size,),
             nprocs=world_size,
             join=True)
if __name__=="__main__":
    main()
