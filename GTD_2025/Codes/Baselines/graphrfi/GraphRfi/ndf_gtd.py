# ndf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
from collections import OrderedDict


class UCIAdultFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate=0., shallow=True):
        super(UCIAdultFeatureLayer, self).__init__()
        self.shallow = shallow
        if shallow:
            self.add_module('linear', nn.Linear(2, 1024))  # 2 = [longitude, latitude]
        else:
            raise NotImplementedError

    def get_out_feature_size(self):
        return 1024


class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class, jointly_training=True):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training

        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).float(), requires_grad=False)

        if jointly_training:
            self.pi = Parameter(torch.rand(self.n_leaf, n_class), requires_grad=True)
        else:
            self.pi = Parameter(torch.ones(self.n_leaf, n_class) / n_class, requires_grad=False)

        self.decision = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_used_feature, self.n_leaf)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, x):
        feats = torch.mm(x, self.feature_mask)
        decision = self.decision(feats)
        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)

        batch_size = x.size(0)
        _mu = x.new_ones(batch_size, 1, 1)
        begin_idx, end_idx = 1, 2
        for _ in range(self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]
            _mu = _mu * _decision
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (_)

        mu = _mu.view(batch_size, self.n_leaf)
        return mu

    def get_pi(self):
        return F.softmax(self.pi, dim=-1) if self.jointly_training else self.pi

    def cal_prob(self, mu, pi):
        return torch.mm(mu, pi)

    def update_pi(self, new_pi):
        self.pi.data = new_pi


class Forest(nn.Module):
    def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training=True):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList([
            Tree(tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training)
            for _ in range(n_tree)
        ])
        self.n_tree = n_tree

    def forward(self, x):
        probs = [tree.cal_prob(tree(x), tree.get_pi()).unsqueeze(2) for tree in self.trees]
        probs = torch.cat(probs, dim=2)
        return probs.mean(dim=2)


class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        out = self.feature_layer(x)
        out = out.view(out.size(0), -1)
        return self.forest(out)

    def loss(self, output, label):
        return self.criterion(torch.log(output), label)
