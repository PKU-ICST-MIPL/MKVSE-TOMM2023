import torch.nn as nn
from torch.nn import Parameter

from lib.util_C_GCN import *


def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, which shared the weight between two separate graphs
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
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
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class C_GCN(nn.Module):
    def __init__(self, text_embeddings, visual_embeddings, adj, opt=None):
        super(C_GCN, self).__init__()
        self.opt = opt
        self.adj = adj
        self.text_embeddings = Parameter(text_embeddings, requires_grad = False) # (2*num_attribute)*300
        self.visual_embeddings = Parameter(l2norm(visual_embeddings,dim=-1), requires_grad = False) # num_attribute*2048
        self.text_linear = nn.Linear(self.text_embeddings.size()[-1], opt.embed_size)
        self.visual_linear = nn.Linear(self.visual_embeddings.size()[-1], opt.embed_size)
        self.gc1 = GraphConvolution(opt.embed_size, opt.embed_size)
        self.gc2 = GraphConvolution(opt.embed_size,  opt.embed_size)
        self.relu = nn.LeakyReLU(0.2)
        self.embed_size = opt.embed_size
        self.init_weights()
        self.adj_Laplacian = Parameter(gen_adj(self.adj), requires_grad = False)

    def init_weights(self):
        """Xavier initialization"""
        r1 = np.sqrt(6.) / np.sqrt(self.text_embeddings.size()[-1] + self.embed_size)
        self.text_linear.weight.data.uniform_(-r1, r1)
        self.text_linear.bias.data.fill_(0)

        r2 = np.sqrt(6.) / np.sqrt(self.visual_embeddings.size()[-1] + self.embed_size)
        self.visual_linear.weight.data.uniform_(-r2, r2)
        self.visual_linear.bias.data.fill_(0)


    def forward(self):
        embeddings = torch.cat([self.text_linear(self.text_embeddings),self.visual_linear(self.visual_embeddings)],dim = 0) # 600*300

        x = self.gc1(embeddings, self.adj_Laplacian)
        x = self.relu(x)
        x = self.gc2(x, self.adj_Laplacian)
        concept_feature = l2norm(x)

        return concept_feature


    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]




