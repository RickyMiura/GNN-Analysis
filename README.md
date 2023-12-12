# Overview
This project is part of my Data Science capstone during my undergraduate program at the University of California, San Diego. In this project, I used Pytorch Geometric to implement different Graph Neural Network (GNN) architectures and evaluated the performance of the models on different graph-structured data. In this repository, I provide the code accompanying my Quarter 1 Project Report which can also be found in the repository. 

# Project Structure
The repository is organized as follows:
```
GNN-Analysis/
├─ GATv2.py
├─ GCN.py
├─ GIN.py
├─ README.md
├─ graphGPS.py
├─ requirements.txt
├─ run.py
```
# Usage
1. Clone this repository on your local machine
2. Open your terminal
3. Change (cd) into the directory to the cloned repository
4. Type  ``` pip install -r requirements.txt```. This contains all the necessary packages for running the code.
5. Use run.py to execute the code. Type ```python run.py {specify model to run}``` in your terminal. Where it says {specify model to run}, replace this with GCN, GAT, GIN, or GPS and only the specified model will be trained and tested on all of the datasets. If no model is specified and you just type ```python run.py```, all of the models will be trained and tested on all of the datasets.

Note that training and testing of modes on the PascalVOC-SP dataset will take ~5 minutes to due to the number of graphs as well as the size of each graph in the dataset.

## Requirements
1) Python 3
2) Libraries listed in requirements.txt

# Contributors
1) Ricky Miura
