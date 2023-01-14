import torch.nn as nn
from torch.nn import Parameter

from lib.util_C_GCN import *


def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class MMGCN_Enc(nn.Module):
    def __init__(self, graph_embed_size, bias=False, dropout = 0.2):
        super(MMGCN_Enc, self).__init__()
        self.dropout = dropout
        self.gc_text = GraphConvolution(graph_embed_size, graph_embed_size, bias=bias)
        self.gc_visual = GraphConvolution(graph_embed_size, graph_embed_size, bias=bias)
        self.gc_all = GraphConvolution(graph_embed_size, graph_embed_size, bias=bias)
        self.relu = nn.LeakyReLU(0.2)


    def forward(self, text_embeddings, visual_embeddings, adj_text, adj_visual, adj_all):
        _x = torch.cat([text_embeddings,visual_embeddings], dim=0)
        # Laplacian Matrix
        _adj_text = gen_adj(adj_text)
        _adj_visual = gen_adj(adj_visual)
        _adj_all = gen_adj(adj_all)

        text_nodes = self.gc_text(text_embeddings, _adj_text)
        visual_nodes = self.gc_visual(visual_embeddings, _adj_visual)
        all_nodes = torch.cat([text_nodes,visual_nodes], dim=0)
        concept_nodes = self.gc_all(all_nodes, _adj_all)
        concept_nodes = l2norm(_x + concept_nodes,dim=-1)
        return concept_nodes
#
#
# class MMGCN_Dec_layer(nn.Module):
#     def __init__(self, graph_embed_size, bias=False, dropout =0.2):
#         super(MMGCN_Dec_layer, self).__init__()
#         self.softmax1 = nn.Softmax(dim=-1)
#         self.score_dec = MMGCN_Enc(1, bias=False)
#         self.softmax2 = nn.Softmax(dim=-1)
#         self.dropout1 = nn.Dropout(p=dropout)
#         self.dropout2 = nn.Dropout(p=dropout)
#
#     def forward(self,query, text_embeddings, visual_embeddings, adj_text, adj_visual, adj_all):
#         _x = query
#         features = torch.cat([text_embeddings, visual_embeddings],dim=0)
#         score = (query @ features.transpose(1, 0)) / math.sqrt(features.size(0))
#         score = self.softmax1(score)
#         score = self.droupout1(score)
#
#
#
#         output= l2norm(_x + score @ features, dim=-1)
#         return output
#
#
# class MMGCN_Dec(nn.Module):
#     def __init__(self, graph_embed_size, bias=False):
#         super(MMGCN_Dec, self).__init__()
#
#
#     def forward(self,query, text_embeddings, visual_embeddings, adj_text, adj_visual, adj_all):
#
#         features = self.mmgcn_enc(text_embeddings, visual_embeddings, adj_text, adj_visual, adj_all)
#
#         score = (query @ features.transpose(1, 0)) / math.sqrt(features.size(0))
#         score = self.softmax1(score)
#         score = self.droupout1(score)
#         score = self.score_dec(score[:,:text_embeddings.size(-1)], score[:,text_embeddings.size(-1):], adj_text, adj_visual, adj_all)
#         score = self.softmax2(score)
#         score = self.droupout2(score)
#         output= l2norm(score @ features, dim=-1)
#         return output
#
#




class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dropout = 0.2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return self.dropout(self.relu(output + self.bias))
        else:
            return self.dropout(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'






