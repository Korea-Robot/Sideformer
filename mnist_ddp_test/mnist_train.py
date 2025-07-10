import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader ,Dataset
import os
import sys
import tempfile
import torch.distributed as dist 
import torch.multiprocessing as mp 

from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP 

def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    
    # 작업 그룹  초기화 
    # dist.init_process_group('gloo',rank=rank,world_size=world_size)
    dist.init_process_group(backend='nccl',rank=rank,world_size=world_size)
    
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
        skp = self.model.state_dict()
        torch.save(skp,"checkpoint.pt")
        print(f"epoch {epoch} | training ckp saved at ckp.pt")
    
    def train(self,max_epochs:int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
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
        shuffle = True
    )


def main(device,total_epochs,save_every):
    dataset,model,optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset,batch_size=1024)
    train = Trainer(model,train_data,optimizer,device,save_every)
    train.train(total_epochs)


if __name__=="__main__":
    import sys
    total_epochs = 100 #int(sys.argv[1])
    save_every   = 10  #  int(sys.argv[2])
    device = 0 # shorthand for cuda:0
    
    # breakpoint()
    main(device,total_epochs,save_every)