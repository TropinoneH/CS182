import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
import os 
import numpy as np
class trainset(Dataset):
    def __init__(self,input_file):
        super().__init__()
        self.data=[]
        self.target=[]
        cl=np.zeros(1)
        count=0
        for genre in (os.listdir(input_file)):
            cl[0]=count
            for file in (os.listdir(os.path.join(input_file,genre))):
                arr=np.load(os.path.join(input_file,genre,file))
                self.data.append(torch.from_numpy(arr).float())
                self.target.append(torch.from_numpy(cl).long())
            cl=np.zeros(1)
            count+=1
    def __getitem__(self, index):
        return self.data[index],self.target[index]
    def __len__(self):
        return len(self.data)

train_dataset = trainset(input_file='./dataset/beat_train')
test_dataset = trainset(input_file='./dataset/beat_test')

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP,self).__init__()
        self.linear1=nn.Linear(5,256)
        self.linear2=nn.Linear(256,256)
        self.linear3=nn.Linear(256,128)
        self.linear4=nn.Linear(128,10)
    
    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=F.relu(self.linear2(x))
        x=F.relu(self.linear3(x))
        x=self.linear4(x)
        return x

def train(Epoch):
    model=MLP().to('cuda:1')
    optimizer=optim.Adam(model.parameters(),lr=0.4,weight_decay=0.99)
    loss_f=nn.CrossEntropyLoss()
    running_loss=0
    for epoch in range (Epoch):
        model.train()
        for i, data in enumerate(train_loader,0):
            in_arr,label=data[0].to('cuda:1'),data[1].to('cuda:1')
            optimizer.zero_grad()
            output=model(in_arr)
            loss=loss_f(output,label.squeeze(1))
            loss.backward()
            optimizer.step()
        running_loss+=loss.item()
        if epoch%100==0:
            print('loss:',running_loss)
        running_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            in_arr,label=data[0].to('cuda:1'),data[1].to('cuda:1')
            output=model(in_arr)
            _,predicted=torch.max(output.data,dim=1)
            _,label_out=torch.max(label,dim=1)
            total+=label_out.size(0)
            correct+=(predicted==label_out).sum().item()
    print('Accuracy:',100*correct/total)


train(1000)