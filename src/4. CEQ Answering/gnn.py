import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn


class HeteroGNNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes,
                 aggregator_type='mean',  # mean pool sum attention?
                 multiple_edge=True,
                 mlp=False,
                 GIN=False,
                 RGCN=False,
                 heads=8,  # if attention
                 relation_reducer='sum',  # sum max s\t\a\c\k
                 rnn=None,  # 'gru',
                 dropout=0.1, norm=False, activation=None):
        super(HeteroGNNLayer, self).__init__()
        # W_r for each relation
        self.GIN = GIN
        if self.GIN:
            aggregator_type = 'sum'
            relation_reducer = 'sum'
            mlp = True
        else:
            self.RGCN = RGCN
            if self.RGCN:
                aggregator_type = 'mean'
                relation_reducer = 'sum'
                mlp = False
                activation = F.relu
        self.aggregator_type = aggregator_type
        self.relation_reducer = relation_reducer
        self.mlp = mlp
        self.dropout = nn.Dropout(dropout)
        if self.aggregator_type == 'attention':
            self.heads = heads
            self.attention_head_size = int(out_size / heads)
            self.all_head_size = self.heads * self.attention_head_size
            if multiple_edge:
                self.fc_query = nn.ModuleDict({name: nn.Linear(in_size, out_size) for name in etypes})
                self.fc_key = nn.ModuleDict({name: nn.Linear(in_size, out_size) for name in etypes})
                self.fc_value = nn.ModuleDict({name: nn.Linear(in_size, out_size) for name in etypes})
            else:
                query_layer = nn.Linear(in_size, out_size)
                key_layer = nn.Linear(in_size, out_size)
                value_layer = nn.Linear(in_size, out_size)
                self.fc_query = nn.ModuleDict({name: query_layer for name in etypes})
                self.fc_key = nn.ModuleDict({name: key_layer for name in etypes})
                self.fc_value = nn.ModuleDict({name: value_layer for name in etypes})
            self.relation_reducer = 'sum'
            # self.fc_out1 = nn.Linear(out_size, 2 * out_size)
            # self.fc_out2 = nn.Linear(2 * out_size, out_size)
        else:
            def gen_mlp():
                return nn.Sequential(
                    nn.Linear(in_size, out_size),
                    # nn.BatchNorm1d(out_size),
                    nn.Dropout(),
                    nn.ReLU(),
                    nn.Linear(out_size, out_size),
                    nn.Dropout(),
                    # nn.BatchNorm1d(out_size),
                    nn.ReLU()
                )

            if multiple_edge:
                if self.mlp:
                    self.fc_trans = nn.ModuleDict({name: gen_mlp() for name in etypes})
                else:
                    self.fc_trans = nn.ModuleDict({name: nn.Linear(in_size, out_size) for name in etypes})
            else:
                if self.mlp:
                    self.fc_trans = nn.ModuleDict({name: gen_mlp() for name in etypes})
                else:
                    trans_layer = nn.Linear(in_size, out_size)
                    self.fc_trans = nn.ModuleDict({name: trans_layer for name in etypes})
        self.fc_self = nn.Linear(in_size, out_size)
        self.fc_neigh = nn.Linear(out_size, out_size)
        if self.relation_reducer == 'stack':
            self.fc_rel = nn.Linear(out_size * len(etypes), out_size)
        self.rnn = None
        if rnn == 'gru':
            self.rnn = nn.GRUCell(out_size, out_size, bias=True)
        if self.GIN:
            self.eps = nn.Parameter(torch.FloatTensor([0]))
        self.activation = activation
        self.norm = nn.LayerNorm(out_size, eps=1e-12) if norm else None
        self.has_fnn = False
        if self.has_fnn:
            self.fnn = nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU(),
                nn.Linear(out_size, in_size),
                nn.Dropout(dropout)
            )
            self.fnn_norm = nn.LayerNorm(out_size, eps=1e-12)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self.aggregator_type == 'attention':
            # nn.init.xavier_uniform_(self.fc_out1.weight, gain=gain)
            # nn.init.xavier_uniform_(self.fc_out2.weight, gain=gain)
            for layer_dict in [self.fc_key, self.fc_query, self.fc_value]:
                for _, layer in layer_dict.items():
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
        else:
            for _, layer in self.fc_trans.items():
                if self.mlp:
                    nn.init.xavier_uniform_(layer[0].weight, gain=gain)
                    nn.init.xavier_uniform_(layer[3].weight, gain=gain)
                else:
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        if self.has_fnn:
            nn.init.xavier_uniform_(self.fnn[0].weight, gain=gain)
            nn.init.xavier_uniform_(self.fnn[2].weight, gain=gain)
        if self.relation_reducer == 'stack':
            nn.init.xavier_uniform_(self.fc_rel.weight, gain=gain)
        if self.rnn is not None:
            self.rnn.reset_parameters()

    def feet_map(self, feat_dict, func):
        feat_dict = {k: func(feat) for k, feat in feat_dict.items()}
        return feat_dict

    def transpose_for_scores(self, x):
        # return x.unsqueeze(1)
        new_x_shape = x.size()[:-1] + (1, self.heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def att_message_func(self, etype):
        def message(edges):
            query = edges.dst['Q_%s' % etype]
            key = edges.dst['K_%s' % etype]
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            return {'V': edges.src['V_%s' % etype], 'S': attention_scores}

        return message

    def att_reduce(self, nodes):
        attention_scores = nodes.mailbox['S']
        alpha = nn.Softmax(dim=1)(attention_scores)
        value = nodes.mailbox['V']
        h = torch.sum(torch.matmul(alpha, value).squeeze(-2), dim=1)
        return {'h': h}

    def forward(self, G, feat_dict, during_train=True):
        # The input is a dictionary of node features for each type
        G = G.local_var()
        # feat_dict = self.feet_map(feat_dict, self.dropout)
        funcs = {}
        h_self = feat_dict
        for srctype, etype, dsttype in G.canonical_etypes:
            if self.aggregator_type == 'attention':
                query = self.transpose_for_scores(self.fc_query[etype](feat_dict[srctype]))
                key = self.transpose_for_scores(self.fc_key[etype](feat_dict[srctype]))
                value = self.transpose_for_scores(self.fc_key[etype](feat_dict[srctype]))
                G.nodes[srctype].data['Q_%s' % etype] = query.float() if during_train else query
                G.nodes[srctype].data['K_%s' % etype] = key.float() if during_train else key
                G.nodes[srctype].data['V_%s' % etype] = value.float() if during_train else value
            else:
                # Compute W_r * h
                Wh = self.fc_trans[etype](feat_dict[srctype])
                if self.aggregator_type == 'pool':
                    Wh = F.relu(Wh)
                # Save it in graph for message passing
                G.nodes[srctype].data['Wh_%s' % etype] = Wh.float()

            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            if self.aggregator_type == 'attention':
                funcs[etype] = (self.att_message_func(etype), self.att_reduce)
            else:
                message = fn.copy_u
                aggregation = fn.mean
                if self.aggregator_type == 'pool':
                    aggregation = fn.max
                elif self.aggregator_type == 'sum':
                    aggregation = fn.sum
                funcs[etype] = (message('Wh_%s' % etype, 'm'), aggregation('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, self.relation_reducer)
        # return the updated node feature dictionary
        h_neigh = {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}
        h_neigh = {k: v.type_as(h_self[k]) for k, v in h_neigh.items()}
        if self.aggregator_type == 'attention':
            h_neigh = {k: v.view(v.size()[:-2] + (self.all_head_size,)) for k, v in h_neigh.items()}
        if self.relation_reducer == 'stack':
            h_neigh = self.feet_map(h_neigh, self.fc_rel)
        if self.rnn is not None:
            rst = {ntype: self.rnn(h_neigh[ntype], h_self[ntype]) for ntype in G.ntypes}
        elif self.GIN:
            rst = {ntype: h_neigh[ntype] + (1 + self.eps) * h_self[ntype] for ntype in G.ntypes}
        else:
            rst = {ntype: self.fc_self(h_self[ntype]) + self.fc_neigh(h_neigh[ntype]) for ntype in G.ntypes}
            if not self.RGCN:
                rst = {ntype: h_self[ntype] + self.dropout(rst[ntype]) for ntype in G.ntypes}
                if self.has_fnn:
                    rst = {ntype: self.fnn_norm(rst[ntype] + self.fnn(rst[ntype])) for ntype in G.ntypes}
        # if self.aggregator_type == 'attention':
        #     intermediate = {ntype: F.relu(self.fc_out1(rst[ntype])) for ntype in G.ntypes}
        #     rst = {ntype: rst[ntype] + self.dropout(self.fc_out2(intermediate[ntype])) for ntype in G.ntypes}
        if self.activation is not None:
            rst = self.feet_map(rst, self.activation)
        # normalization
        if self.norm is not None:
            rst = self.feet_map(rst, self.norm)
        return rst


class HeteroGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_size, n_layers, etypes,
                 dropout=0., aggregator_type='mean', heads=2, norm=False, activation=None, relation_reducer='sum'):
        super(HeteroGNN, self).__init__()
        config = {'etypes': etypes, 'dropout': dropout, 'norm': norm, 'relation_reducer': relation_reducer,
                  'aggregator_type': aggregator_type, 'heads': heads, 'activation': activation}
        # create layers
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(HeteroGNNLayer(in_feats, out_size, **config))
        else:
            self.layers.append(HeteroGNNLayer(in_feats, n_hidden, **config))
            for i in range(1, n_layers - 1):
                self.layers.append(HeteroGNNLayer(n_hidden, n_hidden, **config))
            self.layers.append(HeteroGNNLayer(n_hidden, out_size, **config))

    def forward(self, G, h_dict, during_train=True):
        for layer in self.layers:
            h_dict = layer(G, h_dict, during_train)
            h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        # get logits
        return h_dict


class Model(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        # Use trainable node embeddings as featureless inputs.
        super(Model, self).__init__()
        embed_dict = {ntype: nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        self.gnn = HeteroGNN(in_size, hidden_size, out_size, n_layers=2, etypes=G.etypes)

    def forward(self, G):
        h = self.gnn(G, self.embed)
        return h['paper']


if __name__ == '__main__':
    import scipy.io
    import urllib.request

    data_file_path = '/tmp/ACM.mat'
    # data_url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ACM.mat'
    # urllib.request.urlretrieve(data_url, data_file_path)
    data = scipy.io.loadmat(data_file_path)
    # print(list(data.keys()))

    G = dgl.heterograph({
        ('paper', 'written-by', 'author'): data['PvsA'],
        ('author', 'writing', 'paper'): data['PvsA'].transpose(),
        ('paper', 'citing', 'paper'): data['PvsP'],
        ('paper', 'cited', 'paper'): data['PvsP'].transpose(),
        ('paper', 'is-about', 'subject'): data['PvsL'],
        ('subject', 'has', 'paper'): data['PvsL'].transpose(),
    })
    # print(data['PvsA'])

    print(G)

    pvc = data['PvsC'].tocsr()
    # find all papers published in KDD, ICML, VLDB
    c_selected = [0, 11, 13]  # KDD, ICML, VLDB
    p_selected = pvc[:, c_selected].tocoo()
    # generate labels
    labels = pvc.indices
    labels[labels == 11] = 1
    labels[labels == 13] = 2
    labels = torch.tensor(labels).long()

    # generate train/val/test split
    np.random.seed(42)
    pid = p_selected.row
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:800]).long()
    val_idx = torch.tensor(shuffle[800:900]).long()
    test_idx = torch.tensor(shuffle[900:]).long()

    model = Model(G, 10, 10, 3)

    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(200):
        logits = model(G)
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])

        pred = logits.argmax(1)
        train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
        val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
        test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 5 == 0:
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
            ))
