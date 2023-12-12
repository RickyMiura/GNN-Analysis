#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Imports
import torch
from torch_geometric.nn import GATv2Conv, GPSConv, global_mean_pool, global_add_pool, MLP
from torch_geometric.datasets import Planetoid, TUDataset, LRGBDataset
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout, ModuleList
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import pandas as pd


# In[33]:


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GATv2Conv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Linear
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU)")
else:
    device = torch.device("cpu") 
    print("Using CPU")
    
class GPSConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        prev_channels = in_channels
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(
            Linear(in_channels, 2 * h),
            nn.GELU(),
            Linear(2 * h, h),
            nn.GELU(),
        )
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)
            prev_channels = h
        self.final_conv = GATv2Conv(prev_channels, out_channels, heads=1, dropout=dropout)
        
    def forward(self, x, edge_index):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.final_conv(x, edge_index)
        return F.log_softmax(x, dim=1)


# In[3]:


# Cora Dataset
cora_dataset = Planetoid(root='.', name='Cora')
cora_data = cora_dataset[0].to(device)


# In[4]:


# Train and test GPS on Cora
hidden_channels = [64,64,64]
model = GPSConvNet(cora_dataset.num_node_features, hidden_channels, cora_dataset.num_classes, heads=1, dropout=0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(cora_data.x, cora_data.edge_index)
    loss = F.nll_loss(out[cora_data.train_mask], cora_data.y[cora_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    logits = model(cora_data.x, cora_data.edge_index)
    accs = [torch.sum(logits[mask].argmax(dim=1) == cora_data.y[mask]).item() / mask.sum().item() for mask in [cora_data.train_mask, cora_data.val_mask, cora_data.test_mask]]
    return accs

for epoch in range(50):
    loss = train()
    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test()
        
print(f'Train Accuracy of GPS on Cora Dataset: {train_acc:.4f}')
print(f'Test Accuracy of GPS on Cora Dataset: {test_acc:.4f}')


# In[36]:


# class GPSConvNet(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
#         super(GPSConvNet, self).__init__()
#         self.GPSConvs = nn.ModuleList()
#         prev_channels = in_channels
#         h = hidden_channels[0]
#         self.preprocess = nn.Sequential(
#             Linear(in_channels, 2 * h),
#             nn.GELU(),
#             Linear(2 * h, h),
#             nn.GELU(),
#         )
#         for h in hidden_channels:
#             gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
#             gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
#             self.GPSConvs.append(gps_conv)
#             prev_channels = h
#         self.final_conv = GATv2Conv(prev_channels, out_channels, heads=1, dropout=dropout)

#     def forward(self, x, edge_index, batch):
#         x = x.float() 
#         x = self.preprocess(x)
#         for gps_conv in self.GPSConvs:
#             x = F.relu(gps_conv(x.float(), edge_index))
#             x = F.dropout(x, p=0.6, training=self.training)
#         x = self.final_conv(x, edge_index)
#         x = global_mean_pool(x, batch)
#         return F.log_softmax(x, dim=1)


# In[37]:


# # PascalVOC-SP Dataset
# pasc_dataset = LRGBDataset(root='.', name='PascalVOC-SP')

# train_dataset = pasc_dataset[:int(len(pasc_dataset) * 0.75)]
# test_dataset = pasc_dataset[int(len(pasc_dataset) * 0.75):]

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[38]:


# # Train and test GPS on PascalVOC-SP
# hidden_channels = [64,64,64]
# model = GPSConvNet(pasc_dataset.num_node_features, hidden_channels, pasc_dataset.num_classes, heads=1, dropout=0.0).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(50):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index, data.batch) 
#         loss = F.nll_loss(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#         pred = out.argmax(dim=1)
#         correct += pred.eq(data.y).sum().item()
#         total += data.num_graphs 

#     train_acc = correct / total
#     print(f'Epoch {epoch}, Train Accuracy: {train_acc:.4f}, Loss: {total_loss/len(train_loader)}')


# model.eval() 
# correct = 0
# total = 0

# with torch.no_grad():
#     for data in test_loader:
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.batch)
#         pred = out.argmax(dim=1)
#         correct += pred.eq(data.y).sum().item()
#         total += data.num_graphs

# test_acc = correct / total
# print(f'Test Accuracy: {test_acc:.4f}')


# In[ ]:

# In[ ]:



class GPSConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(
            Linear(in_channels, 2 * h),
            nn.GELU(),
            Linear(2 * h, h),
            nn.GELU(),
        )
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)
            
        self.final_lin = Linear(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index, batch):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)


# In[51]:

from torch_geometric.nn import global_mean_pool 
from sklearn.model_selection import train_test_split
from torch_geometric.nn import global_mean_pool 
from torch_geometric.transforms import OneHotDegree, NormalizeFeatures
from torch_geometric.datasets import TUDataset

hidden_channels = [64, 64, 64]

transform = OneHotDegree(max_degree=135)
dataset = TUDataset(root='.', name='IMDB-BINARY', transform=transform)
data = dataset[0]

model = GPSConvNet(dataset.num_node_features, hidden_channels, dataset.num_classes, heads=1, dropout=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

from torch_geometric.loader import DataLoader

train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum().item())
        total += data.num_graphs
    train_acc = correct / total
    return train_acc
       

def test(loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum().item())
        total += data.num_graphs
    return correct / total

for epoch in range(50):
    loss = train()

loader = DataLoader(dataset, batch_size=32, shuffle=True)
test_acc = test(loader)
print(f'Train Accuracy of GPS on IMDB Dataset: {train_acc:.4f}')
print(f'Test Accuracy of GPS on IMDB Dataset: {test_acc:.4f}')


# In[52]:


# # Enzyme Dataset
# enz_dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)

# loader = DataLoader(enz_dataset, batch_size=32, shuffle=True)


# In[56]:


# class GPSConvNet(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
#         super(GPSConvNet, self).__init__()
#         self.GPSConvs = nn.ModuleList()
#         h = hidden_channels[0]
#         self.preprocess = nn.Sequential(
#             Linear(in_channels, 2 * h),
#             nn.GELU(),
#             Linear(2 * h, h),
#             nn.GELU(),
#         )
#         for h in hidden_channels:
#             gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
#             gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
#             self.GPSConvs.append(gps_conv)

#         # Add a final linear layer for classification
#         self.final_lin = Linear(hidden_channels[-1], out_channels)

#     def forward(self, x, edge_index, batch):
#         x = self.preprocess(x)
#         for gps_conv in self.GPSConvs:
#             x = x.float()
#             x = F.relu(gps_conv(x, edge_index))
#             x = F.dropout(x, p=0.6, training=self.training)
#         x = global_mean_pool(x, batch)
#         x = self.final_lin(x)
#         return F.log_softmax(x, dim=1)


# In[57]:


# # Train and test GPS on Enzyme
# hidden_channels = [64, 64, 64]
# model = GPSConvNet(in_channels=enz_dataset.num_node_features, hidden_channels=hidden_channels, out_channels=enz_dataset.num_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()
# def train():
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     for data in loader:
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index, data.batch)
#         loss = F.nll_loss(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
        
#         pred = out.argmax(dim=1)
#         correct += int((pred == data.y).sum().item())
#         total += data.num_graphs
#     train_acc = correct / total
#     return train_acc

# def test(loader):
#     model.eval()
#     correct = 0
#     total = 0
#     for data in loader:
#         out = model(data.x, data.edge_index, data.batch)
#         pred = out.argmax(dim=1)
#         correct += int((pred == data.y).sum().item())
#         total += data.num_graphs
#     return correct / total

# for epoch in range(50):
#     train_acc = train()

# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# test_acc = test(loader)
# print(f'Train Accuracy of GPS on Enzyme Dataset: {train_acc:.4f}')
# print(f'Test Accuracy of GPS on Enzyme Dataset: {test_acc:.4f}')


# In[ ]:




