import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader ,Dataset
import os
import sys
import tempfile
import torch.distributed as dist 
import torch.multiprocessing as mp  # torch multiprocessing

from torch.utils.data.distributed import DistributedSampler # ddp wrapper data를 gpu로 전ㅐ다


from torch.nn.parallel import DistributedDataParallel as DDP 

## ddp : process start form each gpu
#           each process initialize trainer class object
#           we interest one copy 
def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    
    # 작업 그룹  초기화 
    # dist.init_process_group('gloo',rank=rank,world_size=world_size)
    dist.init_process_group(backend='nccl',rank=rank,world_size=world_size)

    # nvidia collective communication library 
    
    # each gpu run process
    
    # all process discover
    
    # rank  : unique identifier of each process
    # world : total number of processes
    
def cleanup():
    dist.destroy_process_group()
    

# 1. gpu check 

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device ' ,device)


# 2. dataset loading 


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)) # MNIST mean/ std
])

train_dataset  = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset  = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader = DataLoader(train_dataset,batch_size=1024,shuffle=True)
test_loader  = DataLoader(test_dataset,batch_size=64,shuffle=True)





# shimple model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,1), # 28x28 -> 26 x26 
            nn.ReLU(),
            nn.Conv2d(32,64,3,1), # 28x28 -> 26 x26 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(64*12*12,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    
    def forward(self,x):
        return self.net(x)

model = CNN().to(device)

# loss , optimzier 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)


### DDP

class Trainer:
    def __init__(
        self,
        model : torch.nn.Module,
        train_data : DataLoader,
        optimizer  : torch.optim.Optimizer,
        gpu_id :int,
        save_every : int,
    ) -> None:
        self.gpu_id = gpu_id 
        self.model  = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model,device_ids = [self.gpu_id])
    
    def _run_batch(self,source,targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = criterion(output,targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _run_epoch(self,epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | batchsize: {b_sz} | steps {len(self.train_data)}")
        
        for source,targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source,targets)
        print('loss : ',loss)
        
    def _save_checkpoint(self,epoch):
        # skp = self.model.state_dict()
        skp = self.model.module.state_dict()
        torch.save(skp,"checkpoint.pt")
        print(f"epoch {epoch} | training ckp saved at ckp.pt")
    
    def train(self,max_epochs:int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id ==0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                
def load_train_objs():
    train_set = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    return train_set,model,optimizer

def prepare_dataloader(dataset:Dataset,batch_size:int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory = True,
        shuffle = False,             # 반드시 false
        sampler=DistributedSampler(dataset) # ensure data chunk no overapping sample in gpu
    )


def main(rank:int, world_size: int , total_epochs:int,save_every:int,gpu_ids:list): # gpu id add
    setup(rank,world_size) 
    dataset,model,optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset,batch_size=1024)

    # rank(0, 1)를 사용해 실제 GPU 번호(4, 5)를 가져옴
    # rank가 0이면 gpu_ids[0] -> 4
    # rank가 1이면 gpu_ids[1] -> 5
    physical_gpu_id = gpu_ids[rank]
    
    train = Trainer(model,train_data,optimizer,physical_gpu_id,save_every)
    train.train(total_epochs)
    cleanup()
    

if __name__=="__main__":
    from time import time 
    import sys
    start = time()
    total_epochs = 100 #int(sys.argv[1])
    save_every   = 10  #  int(sys.argv[2])
    # device = 0 # shorthand for cuda:0
    gpu_ids = [0,1]
    world_size = len(gpu_ids)
    # mp.spawn(main,args=(world_size,total_epochs,save_every),nprocs= world_size) # 알아서 할당
    mp.spawn(main,args=(world_size,total_epochs,save_every, gpu_ids),nprocs= world_size)

    
    # breakpoint()
    # main(device,total_epochs,save_every)
    
    print()
    print('total time :',(time()-start)//60 ,' MIN')