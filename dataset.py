import os
import dgl
import csv
import time
import torch
import pickle
import random
import numpy as np
import networkx as nx
from dgl import DGLGraph
from dgl.data import citation_graph


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(param):
    if param['percent'] == 0:
        public = 1
    else:
        public = 0
    if param['dataset'] == 'cora':
        data = citation_graph.load_cora(public, param['percent'])

    if param['dataset'] == 'citeseer':
        data = citation_graph.load_citeseer(public, param['percent'])
    if param['dataset'] == 'pubmed':
        data = citation_graph.load_pubmed(public, param['percent'])
      

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)

    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())

    return g, features.to(device), labels.to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device)


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        self.file_path = data_dir + f'/graph_list_labels_{self.split}.pt'
        
        
        if not os.path.isfile(self.file_path):
            with open(data_dir + "/%s.pickle" % self.split, "rb") as f:
                self.data = pickle.load(f)

            with open(data_dir + "/%s.index" % self.split, "r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [self.data[i] for i in data_idx[0]]
                
            assert len(self.data) == num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        
        self.graph_lists = []
        self.graph_labels = []
        self._prepare()
        self.n_samples = len(self.graph_lists)
        
    def _prepare(self):
        if os.path.isfile(self.file_path):
            print(f"load from {self.file_path}")            
            with open(self.file_path, 'rb') as f:
                self.graph_lists, self.graph_labels = pickle.load(f)
            return

        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
        with open(self.file_path, 'wb') as f:
            pickle.dump((self.graph_lists, self.graph_labels), f)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]

