import torch
import torch.nn as nn
from torch import einsum
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import SparseTensor, set_diag


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super(PreNormForward, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super(PreNormAttention, self).__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, share_weight=False, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.share_weight = share_weight
        self.softmax = nn.Softmax(dim=-1)

        self.W_t = nn.Linear(in_features, out_features)
        if share_weight:
            self.W_s = self.W_t
        else:
            self.W_s = nn.Linear(in_features, out_features)
        self.att = nn.Linear(out_features, 1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W_t.weight)
        glorot(self.W_s.weight)
        glorot(self.att.weight)

    def _prepare_attentional_mechanism_input(self, t, s):
        b, nodes, h = t.shape
        q_ = t.repeat(1, nodes, 1)  #[b, n*n, h]
        # print(q_.shape)
        k_ = s.repeat_interleave(nodes, dim=1) #[b, n*n, h]
        print(s.shape)
        print(k_.shape)
        # broadcast add
        e_sum = q_ + k_
        e = e_sum.view(b, nodes, nodes, -1) #[b, n, n, h]
        e = self.leakyrelu(e)
        return e

    def forward(self, t, adj):
        # print(t.shape)
        t_h = self.W_t(t)
        s_h = self.W_s(t)
        e = self._prepare_attentional_mechanism_input(t_h, s_h)
        e = self.att(e)
        e = e.squeeze(-1)
        e = e.masked_fill(adj == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)
        out = torch.matmul(a, s_h)

        if self.concat:
            return F.elu(out)
        else:
            return out + t
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttentionLayer1(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, share_weight=False, concat=False):
        super(GraphAttentionLayer1, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.share_weight = share_weight
        self.softmax = nn.Softmax(dim=-1)

        self.W_t = nn.Linear(in_features, out_features)
        if share_weight:
            self.W_s = self.W_t
        else:
            self.W_s = nn.Linear(in_features, out_features)
        self.att = nn.Linear(out_features, 1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W_t.weight)
        glorot(self.W_s.weight)
        glorot(self.att.weight)

    def _prepare_attentional_mechanism_input(self, q, k):
        b, nodes, h = q.shape
        q_ = q.repeat(1, nodes, 1)  #[b, n*n, h]
        print(q_.shape)
        k_ = k.repeat_interleave(nodes, dim=1) #[b, n*n, h]
        print(k_.shape)
        # broadcast add
        e_sum = q_ + k_
        e = e_sum.view(b, nodes, nodes, -1) #[b, n, n, h]
        e = self.leakyrelu(e)
        return e

    def forward(self, source, target, adj):
        t_h = self.W_t(target)
        s_h = self.W_s(source)
        e = self._prepare_attentional_mechanism_input(t_h, s_h)
        e = self.att(e)
        e = e.squeeze(-1)
        e = e.masked_fill(adj == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)
        out = torch.matmul(a, s_h)

        if self.concat:
            return F.elu(out)
        else:
            return out + target
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(1)  # (b, M, 1)
        beta = torch.softmax(w, dim=1)  # (b, M, 1)
        beta = beta.unsqueeze(1).expand(-1, z.shape[1], -1, -1)  # (b, N, M, 1)
        # print(beta.shape)

        return (beta * z).sum(2)  #  (b, N, M, 1) * (b, n, type, dim) = (b, n, dim)
    
class Resolve_Graph_Layer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, hidden_size, share_weight, concat):
        super(Resolve_Graph_Layer, self).__init__()
        self.trui_graph = GraphAttentionLayer(in_features, out_features, dropout, alpha, share_weight=share_weight, concat=concat)
        self.tril_graph = GraphAttentionLayer(in_features, out_features, dropout, alpha, share_weight=share_weight, concat=concat)
        self.diag_graph = GraphAttentionLayer(in_features, out_features, dropout, alpha, share_weight=share_weight, concat=concat)
        self.l1 = nn.Linear(out_features, out_features)
        self.l2 = nn.Linear(out_features, out_features)
        self.l3 = nn.Linear(out_features, out_features)

    def forward(self, source, adj_list):
        diag_graph = self.trui_graph(source, adj_list[0])
        trui_graph = self.tril_graph(source, adj_list[1])
        tril_graph = self.diag_graph(source, adj_list[2])
        output = self.l1(diag_graph) + self.l2(trui_graph) + self.l3(tril_graph)

        return output

class Dispersed_Graph_Layer(nn.Module):
    def __init__(self, in_features, dropout, alpha, hidden_size, share_weight=False, concat=False):
        super(Dispersed_Graph_Layer, self).__init__()
        self.hidden_dim = int(in_features / 2)
        self.graph_layer = nn.ModuleList([])
        for _ in range(2):
            self.graph_layer.append(Resolve_Graph_Layer(self.hidden_dim, self.hidden_dim, dropout, alpha, hidden_size, 
                                                        share_weight=share_weight, concat=concat))

    def forward(self, source, adj_list):
        b, n, h = source.shape
        hidden_dim = int(h / 2)
        s = source.view(b, -1, n, hidden_dim)  # [batch, 2, node, hidden/2]
        # t = target.view(b, -1, n, hidden_dim)
        output = []
        for i in range(2):
            output1 = self.graph_layer[i](s[:, i], adj_list)
            # print(self.graph_layer[i])
            output.append(output1)
        return output

class Dispersed_Graph(nn.Module):
    def __init__(self, depth, in_features, dropout, alpha, hidden_size, share_weight=False, concat=False):
        super(Dispersed_Graph, self).__init__()
        self.depth = depth
        # self.fist_l = Resolve_Graph_Layer(in_features[0], in_features[0], dropout=dropout, alpha=alpha, 
        #                                   hidden_size=hidden_size, share_weight=share_weight, concat=concat)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layer = nn.ModuleList([])
            for j in range(2**i):
                self.layer.append(Dispersed_Graph_Layer(in_features[i], dropout, alpha, hidden_size, 
                                                        share_weight=share_weight, concat=concat))
            self.layer.append(nn.Linear(int(in_features[i]/2), int(in_features[i]/2), bias=False))
            self.layers.append(self.layer)

    def forward(self, target,  adj_list):
        num_layers = self.depth
        i = 0
        # target = self.fist_l(source_list[-1], target, adj_list)
        for layer in self.layers:
            output = []
            if i==0:
                for j in range(2**i):
                # 1   t:512   s:512 ->  ts [2, 256]
                    t = layer[j](target, adj_list)
                    output.append(layer[2**i](t[0]))
                    output.append(layer[2**i](t[1]))
                target = output
            else:
                target_ = target
                for j in range(2**i):
                    t = layer[j](target_[j], adj_list)
                    # print(layer[j])
                    output.append(layer[2**i](t[0]))
                    output.append(layer[2**i](t[1]))
                    # output.append(t[0])
                    # output.append(t[1])
                target = output
            i += 1
        target = torch.stack(target, dim=-1)
        target = target.view(target.size(0), target.size(1), -1)
        return target

class Dispersed(nn.Module):
    def __init__(self, layer_num, depth, in_features, dropout, alpha, hidden_size, share_weight=False, concat=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(layer_num):
            self.layers.append(nn.ModuleList([
                Dispersed_Graph(depth, in_features, dropout, alpha, hidden_size, share_weight=share_weight, concat=concat),
                FeedForward(in_features[0], in_features[0], dropout=dropout)
            ]))
    
    def forward(self, source, target, adj_list):
        for graph, fft in range(self.layers):
            target = graph(source, target, adj_list) + target
            target = fft(target) + target
        return target

class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads=8, dim_head=64, dropout=0.3):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v, mask=None):

        b, n, _, h = *q.shape, self.heads
        # print(q.shape)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            dots = mask * dots
            attn = self.attend(dots)
        else:
            attn = self.attend(dots)
            # print(attn[0, 0, 0, :])
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print(out.shape)

        return self.to_out(out)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.att = PreNormAttention(dim, Attention(dim, dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ffn = PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x, y, mask=None):
        att = self.att(x, y, y, mask=mask) + x
        ffn = self.ffn(att) + att

        return ffn

class CrossTransEncoder(nn.Module):
    def __init__(self, dim, out_dim, heads, dim_head, mlp_dim, depth, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerEncoderLayer(dim[i],  heads[i], dim_head, mlp_dim[i], dropout=dropout),
                nn.Linear(dim[i], out_dim[i]),
                TransformerEncoderLayer(dim[i],  heads[i], dim_head, mlp_dim[i], dropout=dropout),
                nn.Linear(dim[i], out_dim[i])
            ]))
    
    def forward(self, x, y, mask=None, save_hidden=False):
        if save_hidden==True:
            x_list = []
            x_list.append(x)
            y_list = []
            y_list.append(y)
            for att1, l1, att2, l2 in self.layers:
                x_ = x
                y_ = y
                x = att1(x_, y_, mask=mask)
                x = l1(x)
                y = att2(y_, x_, mask=mask)
                y = l2(y)
                x_list.append(x)
                y_list.append(y)
            return x_list, y_list
        else:
            for att1, l1, att2, l2 in self.layers:
                x_ = x
                y_ = y
                x = att1(x_, y_, mask=mask)
                x = l1(x)
                y = att2(y_, x_, mask=mask)
                y = l2(y)
            return x, y

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, y, mask=None, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, y, y, mask=mask) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, y, y, mask=mask) + x
                x = ff(x) + x
            return x

class TransformerEncoder_ver1(nn.Module):
    def __init__(self, in_dim, out_dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super(TransformerEncoder_ver1, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(in_dim[i], Attention(in_dim[i], out_dim[i], heads=heads[i], dim_head=dim_head, dropout=dropout)),
                nn.Linear(in_dim[i], out_dim[i]),
                PreNormForward(out_dim[i], FeedForward(out_dim[i], mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            i = 0
            for attn, linear, ff in self.layers:
                x_ = linear(x)
                x = attn(x, x, x, mask=mask) + x_
                x = ff(x) + x
                hidden_list.append(x)
                i += 1
            return hidden_list
        else:
            i = 0
            for attn, linear, ff in self.layers:
                x_ = linear(x)
                x = attn(x, x, x, mask=mask) + x_
                x = ff(x) + x
                i += 1
            return x
        
class TransformerEncoder_ver2(nn.Module):
    def __init__(self, in_dim, out_dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super(TransformerEncoder_ver2, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(in_dim[i], Attention(in_dim[i], out_dim[i], heads=heads[i], dim_head=dim_head, dropout=dropout)),
                nn.Linear(in_dim[i], out_dim[i]),
                PreNormForward(out_dim[i], FeedForward(out_dim[i], mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, y, mask=None, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            i = 0
            for attn, linear, ff in self.layers:
                x_ = linear(x)
                x = attn(x, y[i], y[i], mask=mask) + x_
                x = ff(x) + x
                hidden_list.append(x)
                i += 1
            return hidden_list
        else:
            i = 0
            for attn, linear, ff in self.layers:
                x_ = linear(x)
                x = attn(x, y[i], y[i], mask=mask) + x_
                x = ff(x) + x
                i += 1
            return x