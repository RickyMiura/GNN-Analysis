#!/usr/bin/env python
# coding: utf-8

# In[61]:


# Imports
import torch
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, MLP
from torch_geometric.datasets import Planetoid, TUDataset, LRGBDataset
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import pandas as pd


# In[65]:


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# In[66]:


# Cora Dataset
cora_dataset = Planetoid(root='.', name='Cora')
cora_data = cora_dataset[0]


# In[67]:


# Train and test GCN on Cora
model = GCN(cora_dataset.num_node_features, cora_dataset.num_classes)
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
print(f'Train Accuracy of GCN on Cora Dataset: {train_acc:.4f}')
print(f'Test Accuracy of GCN on Cora Dataset: {test_acc:.4f}')


# In[68]:


# PascalVOC-SP Dataset
pasc_dataset = LRGBDataset(root='.', name='PascalVOC-SP')

train_dataset = pasc_dataset[:int(len(pasc_dataset) * 0.8)]
test_dataset = pasc_dataset[int(len(pasc_dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[69]:


# Train and test GCN on PascalVOC-SP
model = GCN(num_features=pasc_dataset.num_node_features, num_classes=pasc_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
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
print(f'Train Accuracy of GCN on PascalVOC-SP Dataset: {train_acc:.4f}')

# Testing loop
model.eval()
correct = 0
total = 0
for data in test_loader:
    out = model(data)
    pred = out.argmax(dim=1)
    correct += pred.eq(data.y).sum().item()
    total += data.num_nodes

test_acc = correct / total
print(f'Test Accuracy of GCN on PascalVOC-SP Dataset: {test_acc:.4f}')


# In[30]:


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.out(x)       
        return x


# In[39]:


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


# In[40]:


# Train and test GCN on IMDB
model = GCN(num_node_features=modified_imdb_dataset[0].num_features, num_classes=len(set([data.y.item() for data in modified_imdb_dataset])), hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    model.train()
    train_corr = 0
    total = 0

    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        train_corr += int((pred == data.y).sum())
        total += data.y.size(0)

    train_acc = train_corr / total
print(f'Train Accuracy of GCN on IMDB Dataset: {train_acc:.4f}')

model.eval()
test_corr = 0
total = 0

for data in loader: 
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1)
    test_corr += int((pred == data.y).sum())
    total += data.y.size(0)

test_acc = test_corr / total
print(f'Test Accuracy of GCN on IMDB Dataset: {test_acc:.4f}')


# # In[41]:


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.bn1 = BatchNorm1d(64)
        self.conv2 = GCNConv(64, 32)
        self.bn2 = BatchNorm1d(32)
        self.lin = Linear(32, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


# In[45]:


# Enzyme Dataset
enz_dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)

loader = DataLoader(enz_dataset, batch_size=32, shuffle=True)


# In[46]:


# Train and test GCN on Enzyme
model = GCN(num_node_features=enz_dataset.num_node_features, num_classes=enz_dataset.num_classes)
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
print(f'Train Accuracy of GCN on Enzyme Dataset: {train_acc:.4f}')

model.eval()
test_corr = 0
for data in loader: 
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1) 
    test_corr += int((pred == data.y).sum())
test_acc = test_corr / len(loader.dataset)
print(f'Test Accuracy of GCN on Enzyme Dataset: {test_acc:.4f}')


# In[ ]:




