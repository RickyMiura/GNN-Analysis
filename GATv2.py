#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import torch
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, MLP
from torch_geometric.datasets import Planetoid, TUDataset, LRGBDataset
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import pandas as pd


# In[2]:


class GATv2(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(8 * 8, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# In[3]:


# Cora Dataset
cora_dataset = Planetoid(root='.', name='Cora')
cora_data = cora_dataset[0]


# In[4]:


# Train and test GAT on Cora
model = GATv2(cora_dataset.num_node_features, cora_dataset.num_classes)
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
print(f'Train Accuracy of GAT on Cora Dataset: {train_acc:.4f}')
print(f'Test Accuracy of GAT on Cora Dataset: {test_acc:.4f}')


# In[23]:


# PascalVOC-SP Dataset
pasc_dataset = LRGBDataset(root='.', name='PascalVOC-SP')

train_dataset = pasc_dataset[:int(len(pasc_dataset) * 0.8)]
test_dataset = pasc_dataset[int(len(pasc_dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[24]:


# Train and test GAT on PascalVOC-SP
model = GATv2(num_features=pasc_dataset.num_node_features, num_classes=pasc_dataset.num_classes)
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
print(f'Train Accuracy of GAT on PascalVOC-SP Dataset: {train_acc:.4f}')

model.eval()
correct = 0
total = 0
for data in test_loader:
    out = model(data)
    pred = out.argmax(dim=1)
    correct += pred.eq(data.y).sum().item()
    total += data.num_nodes

test_acc = correct / total
print(f'Test Accuracy of GAT on PascalVOC-SP Dataset: {test_acc:.4f}')


# # In[10]:


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


# In[11]:


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.out = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.out(x)
        x  = F.log_softmax(x, dim=1)
        return x


# In[13]:


# Train and test GAT on IMDB
model = GAT(num_node_features=modified_imdb_dataset[0].num_features, num_classes=len(set([data.y.item() for data in modified_imdb_dataset])), hidden_channels=8)
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
print(f'Train Accuracy of GAT on IMDB Dataset: {train_acc:.4f}')

model.eval()
test_corr = 0
for data in loader: 
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1) 
    test_corr += int((pred == data.y).sum())
test_acc = test_corr / len(loader.dataset)
print(f'Test Accuracy of GAT on IMDB Dataset: {test_acc:.4f}')


# # In[14]:


# # Enzyme Dataset
enz_dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)

loader = DataLoader(enz_dataset, batch_size=32, shuffle=True)


# In[15]:


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


# In[17]:


# Train and test GAT on Enzyme
model = GAT(in_channels=enz_dataset.num_node_features, hidden_channels=32, out_channels=enz_dataset.num_classes)
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
print(f'Train Accuracy of GAT on Enzyme Dataset: {train_acc:.4f}')

model.eval()
test_corr = 0
for data in loader: 
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1) 
    test_corr += int((pred == data.y).sum())
test_acc = test_corr / len(loader.dataset)
print(f'Test Accuracy of GAT on Enzyme Dataset: {test_acc:.4f}')


# In[ ]:




