import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.data import Data
from graph_conv import GraphConv
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from sklearn.metrics import  f1_score

import pdb
import numpy as np
import random
import copy
import argparse

from dataset import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser('Mixup')
parser.add_argument('--dataset', type=str, default="pubmed_raw")
parser.add_argument('--mixup', type=str, default="False", help='Whether to have Mixup')
parser.add_argument('--imb_class', type=str, default="01")
parser.add_argument('--imb_ratio', type=float, default=0.2)
parser.add_argument('--seed', type=float, default=1033)
#parser.add_argument('--dataset', type=str, default="Pubmed")

args = parser.parse_args()

int_list=[]
for char in args.imb_class:
    int_list.append(int(char))
args.imb_class=int_list
print(args.imb_class)

def idNode(data, id_new_value_old):
    data = copy.deepcopy(data)
    data.x = None
    data.y[data.val_id] = -1
    data.y[data.test_id] = -1
    data.y = data.y[id_new_value_old]

    data.train_id = None
    data.test_id = None
    data.val_id = None

    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype = torch.long).to(device)
    id_old_value_new[id_new_value_old] = torch.arange(0, id_new_value_old.shape[0], dtype = torch.long).to(device)
    row = data.edge_index[0].to(device)
    col = data.edge_index[1].to(device)
    row = id_old_value_new[row]
    col = id_old_value_new[col]
    data.edge_index = torch.stack([row, col], dim=0)

    return data

def shuffleData(data):
    data = copy.deepcopy(data)
    id_new_value_old = np.arange(data.num_nodes)
    train_id_shuffle = copy.deepcopy(data.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data.train_id.cpu().numpy()] = train_id_shuffle.cpu().numpy()
    data = idNode(data, id_new_value_old)

    return data, id_new_value_old


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(Net, self).__init__()
        self.conv1 = GraphConv(in_channel, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(1 * hidden_channels, out_channel)

    def forward(self, x0, edge_index, edge_index_b, lam, id_new_value_old):

        x1 = self.conv1(x0, edge_index, x0)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x2 = self.conv2(x1, edge_index, x1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)
        
        x0_b = x0[id_new_value_old]
        x1_b = x1[id_new_value_old]
        x2_b = x2[id_new_value_old]

        x0_mix = x0 * lam + x0_b * (1 - lam)

        new_x1 = self.conv1(x0, edge_index, x0_mix)
        new_x1_b = self.conv1(x0_b, edge_index_b, x0_mix)
        new_x1 = F.relu(new_x1)
        new_x1_b = F.relu(new_x1_b)

        x1_mix = new_x1 * lam + new_x1_b * (1 - lam)
        x1_mix = F.dropout(x1_mix, p=0.4, training=self.training)

        new_x2 = self.conv2(x1, edge_index, x1_mix)
        new_x2_b = self.conv2(x1_b, edge_index_b, x1_mix)
        new_x2 = F.relu(new_x2)
        new_x2_b = F.relu(new_x2_b)

        x2_mix = new_x2 * lam + new_x2_b * (1 - lam)
        x2_mix = F.dropout(x2_mix, p=0.4, training=self.training)

        new_x3 = self.conv3(x2, edge_index, x2_mix)
        new_x3_b = self.conv3(x2_b, edge_index_b, x2_mix)
        new_x3 = F.relu(new_x3)
        new_x3_b = F.relu(new_x3_b)

        x3_mix = new_x3 * lam + new_x3_b * (1 - lam)
        x3_mix = F.dropout(x3_mix, p=0.4, training=self.training)

        x = x3_mix
        x = self.lin(x)
        return x.log_softmax(dim=-1)


# set random seed
# SEED = 1033
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
# np.random.seed(SEED)  # Numpy module.
# random.seed(SEED)  # Python random module.
seed_everything(seed=1033)

'''
# load data
dataset = "Pubmed"
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]


# split data
node_id = np.arange(data.num_nodes)
np.random.shuffle(node_id)
# data.train_id = node_id[:int(data.num_nodes * 0.6)]
# data.val_id = node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)]
# data.test_id = node_id[int(data.num_nodes * 0.8):]

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


indices = []
    
num_classes = data.y.max().item() + 1
for i in range(num_classes):
    index = torch.nonzero(data.y == i).view(-1)
    index = index[torch.randperm(index.size(0))]
    indices.append(index)

#print(indices)
train_index = []
res_index = []
num_train_node=20
for i, _ in enumerate(indices):
    #print(i,_)
    if i not in args.imb_class:
        train_index.append(_[:num_train_node])
        res_index.append(_[num_train_node:])
    else:
        train_index.append(_[:int(num_train_node*args.imb_ratio)])
        res_index.append(_[int(num_train_node*args.imb_ratio):])

train_index = torch.cat(train_index, dim=0)
res_index = torch.cat(res_index, dim=0)

res_index = res_index[torch.randperm(res_index.size(0))]

data.train_mask = index_to_mask(train_index, size=data.num_nodes)
data.val_mask = index_to_mask(res_index[:500], size=data.num_nodes)
data.test_mask = index_to_mask(res_index[500:], size=data.num_nodes)

data.train_id = node_id[data.train_mask]
data.val_id = node_id[data.val_mask ]
data.test_id = node_id[data.test_mask]

print(data.train_id.shape)
'''

# load data
data, split_edge, args = get_dataset(args.dataset, args)



#print(data.train_id.shape,data.val_id.shape, data.test_id.shape) #torch.Size([28]) torch.Size([500]) torch.Size([19189])

# define model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(dataset.num_node_features, args.num_features) # 500 1433
# print(dataset.num_classes, args.num_classes) # 3 3
# model = Net(hidden_channels=64, in_channel = dataset.num_node_features, out_channel = dataset.num_classes).to(device)
model = Net(hidden_channels=64, in_channel = args.num_features, out_channel = args.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# func train one epoch
def train(data):
    model.train()

    if args.mixup=="True":
        lam = np.random.beta(4.0, 4.0)
    else:
        print("not mixup")
        lam = 1.0

    data_b, id_new_value_old = shuffleData(data)
    data = data.to(device)
    data_b = data_b.to(device)

    optimizer.zero_grad()

    out = model(data.x, data.edge_index, data_b.edge_index, lam, id_new_value_old)
    loss = F.nll_loss(out[data.train_id], data.y[data.train_id]) * lam + \
           F.nll_loss(out[data.train_id], data_b.y[data.train_id]) * (1 - lam)

    loss.backward()
    optimizer.step()

    return loss.item()


# test
@torch.no_grad()
def test(data):
    model.eval()

    out = model(data.x.to(device), data.edge_index.to(device), data.edge_index.to(device), 1, np.arange(data.num_nodes))
    # pred = out.argmax(dim=-1)
    # y_pred = pred.cpu().numpy()
    # y_true=data.y.to(device)
    
    # correct = pred.eq(y_true)

    accs = []
    f1s=[]
    for _, id_ in data('train_id', 'val_id', 'test_id'):
        # accs.append(correct[id_].sum().item() / id_.shape[0])
        # f1s.append(f1_score(y_true[id_], y_pred, average='macro'))
        pred = out[id_].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data.y[id_].cpu().numpy()
        acc = pred.eq(data.y[id_]).sum().item() / id_.shape[0]
        f1 = f1_score(y_true, y_pred, average='macro')

        accs.append(acc)
        f1s.append(f1)
        
    
    
    min_mask= torch.zeros(size=data.y.shape).to(device)
    #print(data_test_mask.shape)
    for i in args.imb_class:
        min_mask= ((data.y == i).bool() | min_mask.bool()).to(device)
    
    maj_mask= ~min_mask
    
    min_mask=min_mask & data.test_mask
    maj_mask=maj_mask & data.test_mask
    #print(maj_mask.shape)
        
    maj_pred = out[maj_mask].max(1)[1]
    maj_acc =maj_pred.eq(data.y[maj_mask]).sum().item() / maj_mask.sum().item()
    
    min_pred = out[min_mask].max(1)[1]
    min_acc = min_pred.eq(data.y[min_mask]).sum().item() / min_mask.sum().item()
    
        
    
    return accs,f1s,maj_acc, min_acc

final_accs=[]
final_f1s=[]
final_diffs=[]
for run in range(5):
    seed_everything(args.seed+run)
    best_val_acc = test_acc = best_val_f1 = test_f1= test_min_acc=test_maj_acc=0
    for epoch in range(1,1000):
        data=data.to(device)
        loss = train(data)
        accs, f1s, maj_acc, min_acc= test(data)
        train_acc, val_acc, tmp_test_acc = accs
        train_f1, tmp_val_f1, tmp_test_f1 = f1s
        #print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {accs[0]:.4f}, Test Acc: {accs[2]:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # val_f1 = tmp_val_f1
            test_acc = tmp_test_acc
            test_f1 = f1s[2]
            test_min_acc=min_acc
            test_maj_acc= maj_acc
    
    final_accs.append(test_acc) 
    final_f1s.append(test_f1)  
    final_diffs.append(test_maj_acc-test_min_acc)     
    print(f"Best Test accuracy:{test_acc}, Best Test f1: {test_f1}, Maj-Min: {test_maj_acc-test_min_acc}" )
    print(final_accs)
import statistics 
# print(accs)
# print(f1s)
print(f"Ave Acc: {sum(final_accs)/len(final_accs)} +- {statistics.pstdev(final_accs)}, Ave f1: {sum(final_f1s)/len(final_f1s)} +- {statistics.pstdev(final_f1s)}, Maj-Min: {sum(final_diffs)/len(final_diffs)}")