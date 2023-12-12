#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import torch
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, MLP
from torch_geometric.datasets import Planetoid, TUDataset, LRGBDataset
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import pandas as pd


# In[2]:


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, 16), 
            torch.nn.ReLU(), 
            torch.nn.Linear(16, 16)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(16)
        nn2 = torch.nn.Sequential(torch.nn.Linear(16, num_classes))
        self.conv2 = GINConv(nn2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# In[3]:


# Cora Dataset
cora_dataset = Planetoid(root='.', name='Cora')
cora_data = cora_dataset[0]


# In[4]:


# Train and test GIN on Cora
model = GIN(cora_dataset.num_node_features, cora_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(cora_data)
    loss = F.nll_loss(out[cora_data.train_mask], cora_data.y[cora_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    logits, accs = model(cora_data), []
    for _, mask in cora_data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(cora_data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test()
print(f'Train Accuracy of GIN on Cora Dataset: {train_acc:.4f}')
print(f'Test Accuracy of GIN on Cora Dataset: {test_acc:.4f}')


# In[5]:


# PascalVOC-SP Dataset
pasc_dataset = LRGBDataset(root='.', name='PascalVOC-SP')

train_dataset = pasc_dataset[:int(len(pasc_dataset) * 0.8)]
test_dataset = pasc_dataset[int(len(pasc_dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[7]:


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIN, self).__init__()
        mlp1 = MLP([num_features, 32, 32], batch_norm=True)
        mlp2 = MLP([32, 32, num_classes], batch_norm=True)

        self.conv1 = GINConv(mlp1, train_eps=True)
        self.conv2 = GINConv(mlp2, train_eps=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# In[8]:


# Train and test GIN on PascalVOC-SP
model = GIN(num_features=pasc_dataset.num_node_features, num_classes=pasc_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_nodes

    train_acc = correct / total
print(f'Train Accuracy of GIN on PascalVOC-SP Dataset: {train_acc:.4f}')

model.eval()
correct = 0
total = 0
for data in test_loader:
    out = model(data)
    pred = out.argmax(dim=1)
    correct += pred.eq(data.y).sum().item()
    total += data.num_nodes

test_acc = correct / total
print(f'Test Accuracy of GIN on PascalVOC-SP Dataset: {test_acc:.4f}')


# In[9]:


# IMDB Dataset
imdb_dataset = TUDataset(root='.', name='IMDB-BINARY')

# Add degree of a node as node feature
modified_imdb_dataset = []
for data in imdb_dataset:
    num_nodes = data.num_nodes
    constant_feature = torch.ones((num_nodes, 1))
    edge_index = data.edge_index
    deg = degree(edge_index[0], num_nodes).view(-1, 1).float()
    data.x = torch.cat([constant_feature, deg], dim=1)
    modified_imdb_dataset.append(data)
    
loader = DataLoader(modified_imdb_dataset, batch_size=32, shuffle=True)


# In[10]:


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(GIN, self).__init__()
        mlp1 = Sequential(
            Linear(2, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels), 
            BatchNorm1d(hidden_channels)
        )
        self.conv1 = GINConv(mlp1)
        mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels), 
            BatchNorm1d(hidden_channels)
        )
        self.conv2 = GINConv(mlp2)
        self.lin = Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


# # In[12]:


# Train and test GIN on IMDB
model = GIN(hidden_channels=64, num_classes=len(set([data.y.item() for data in modified_imdb_dataset])))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    model.train()
    train_corr = 0
    total = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred = out.argmax(dim=1)
        train_corr += int((pred == data.y).sum())
        total += data.y.size(0)

    train_acc = train_corr / total
print(f'Train Accuracy of GIN on IMDB Dataset: {train_acc:.4f}')

model.eval()
test_corr = 0
for data in loader: 
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1) 
    test_corr += int((pred == data.y).sum())
test_acc = test_corr / len(loader.dataset)
print(f'Test Accuracy of GIN on IMDB Dataset: {test_acc:.4f}')


# # In[13]:


# Enzyme Dataset
enz_dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)

loader = DataLoader(enz_dataset, batch_size=32, shuffle=True)


# In[14]:


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.mlp1 = Sequential(
            Linear(in_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels), 
            BatchNorm1d(hidden_channels)
        )
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels)
        )
        self.conv2 = GINConv(self.mlp2)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


# In[15]:


# Train and test GIN on Enzyme
model = GIN(in_channels=enz_dataset.num_node_features, hidden_channels=64, out_channels=enz_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    model.train()
    train_corr = 0
    total = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pred = out.argmax(dim=1)
        train_corr += int((pred == data.y).sum())
        total += data.y.size(0)

    train_acc = train_corr / total
print(f'Train Accuracy of GIN on Enzyme Dataset: {train_acc:.4f}')

model.eval()
test_corr = 0
for data in loader: 
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1) 
    test_corr += int((pred == data.y).sum())
test_acc = test_corr / len(loader.dataset)
print(f'Test Accuracy of GIN on Enzyme Dataset: {test_acc:.4f}')


# In[ ]:




