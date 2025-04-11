import os
import numpy as np
import nni
import csv
import json
import time
import warnings
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import get_model
from dataset import load_data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 7
np.random.seed(seed)
tf.random.set_seed(seed)

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, mode='test')
        logits = logits[mask]
        labels = labels[mask]
        _, pred = torch.max(logits, dim=1)
        correct = torch.sum(pred == labels)
        return correct.item() * 1.0 / len(labels)

# 3D plots
def ShowPlot(x, y):
    tsne = TSNE(n_components=3, verbose=1, random_state=123)
    z = tsne.fit_transform(x)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    df["comp-3"] = z[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xs=df["comp-1"], ys=df["comp-2"], zs=df["comp-3"], c=df["y"], cmap="viridis")
    ax.set_title("cora")
    ax.set_xlabel("Hyper feature 1")
    ax.set_ylabel("Hyper feature 2")
    ax.set_zlabel("Hyper feature 3")
    fig.colorbar(scatter)

    plt.show()
#model-summary
def model_summary(model):
    print(model)
    print("Model Summary:")
    print("-" * 50)
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_params += parameter.numel()
        total_params += parameter.numel()
        print(f"{name}\t\t{parameter.shape}\t\t{parameter.numel()}\t\t{parameter.requires_grad}")
    print("-" * 50)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {total_params - trainable_params}")
    print("-" * 50)

def main(param):
    g, features, labels, train_mask, val_mask, test_mask = load_data(param)
    param['input_dim'] = features.shape[1]
    param['output_dim'] = torch.max(labels) + 1
    model = get_model(param).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    test_best = 0
    test_val = 0
    val_best = 0
    val_best_epoch = 0
    train_loss1 = []
    train_acc1 = []
    val_acc1 = []
    loss_11 = []
    loss_22 = []
    loss_33 = []

    for epoch in range(param['epochs']):
        model.train()
        optimizer.zero_grad()
        model.g = g
        logits = model(features)
        pred = F.log_softmax(logits, 1)
        loss_cla = F.nll_loss(pred[train_mask], labels[train_mask])
        loss_graph, loss_node = model.compute_disentangle_loss()
        loss = loss_cla + loss_graph * param['ratio_graph'] + loss_node * param['ratio_node']
        loss.backward()
        optimizer.step()
        loss_1 = loss_cla.item()
        loss_2 = loss_graph.item() * param['ratio_graph']
        loss_3 = loss_node.item() * param['ratio_node']
        loss_11.append(loss_1)
        loss_22.append(loss_2)
        loss_33.append(loss_3)
        train_loss = loss.item()
        train_acc = evaluate(model, features, labels, train_mask)
        val_acc = evaluate(model, features, labels, val_mask)
        test_acc = evaluate(model, features, labels, test_mask)
        train_loss1.append(train_loss)
        train_acc1.append(train_acc)
        val_acc1.append(val_acc)

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            val_best_epoch = epoch
        print("\033[0;30;46m Epoch: {} | Loss: {:.6f} | Acc: {:.5f}, {:.5f}".format(epoch, loss_1, val_acc, train_acc))

    model_summary(model)  # Print model summary

    ShowPlot(features, labels)

    # Show plots
    plt.figure(figsize=(12, 6))
    plt.plot(train_acc1, label='train acc')
    plt.plot(val_acc1, label='val acc')
    plt.legend()
    plt.title('accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    input()
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss1, label='train loss')
    plt.plot(loss_11, label='loss 1')
    plt.legend()
    plt.title('loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--ExpName", type=str, default='run0000')
    parser.add_argument("--model", type=str, default='MNGCN')
    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument("--input_dim", type=int, default=30)
    parser.add_argument("--out_dim", type=int, default=6)
    parser.add_argument("--percent", type=float, default=0.03)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--ablation_mode", type=int, default=0)

    parser.add_argument("--num_graph", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=18)
    parser.add_argument("--graph_dim", type=int, default=18)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--ratio_graph", type=float, default=1.0)
    parser.add_argument("--ratio_node", type=float, default=1.0)
    parser.add_argument("--num_hop", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--graph_size', type=int, default=30)
    parser.add_argument('--graph_num', type=int, default=10000)
    parser.add_argument('--feature_num', type=int, default=5)
    parser.add_argument('--std', type=float, default=5.0)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--init', type=float, default=0.2)
    parser.add_argument('--selected_num', type=int, default=5)

    args = parser.parse_args()
    print(args)
    if args.dataset == 'cora':
        jsontxt = open("./param/param_cora.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'citeseer':
        jsontxt = open("./param/param_citeseer.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'pubmed':
        jsontxt = open("./param/param_pubmed.json", 'r').read()
        param = json.loads(jsontxt)
    else:
        param = args.__dict__

    param.update(nni.get_next_parameter())
    print(param)

    if args.dataset == '':
        main(param)

    else:
        main(param)
        
