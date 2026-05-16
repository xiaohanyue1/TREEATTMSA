import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter
from torch_geometric.utils import softmax, degree
from torch_geometric.nn.conv import MessagePassing
#from torch_geometric.nn import aggr
from torch_geometric.nn.inits import glorot, zeros


class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads, use_attention=False,
                 concat=True, negative_slope=0.2, dropout=0.3, bias=True):
        super(HypergraphConv, self).__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.heads = heads if use_attention else 1
        self.concat = concat
        self.act = nn.LeakyReLU()
        self.dense = nn.Linear(self.heads * out_channels, in_channels)
        # self.dense = nn.Linear(in_channels, self.heads * out_channels)
        # 可训练参数
        self.weight = Parameter(torch.Tensor(in_channels, self.heads * out_channels))
        if use_attention:
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
            self.negative_slope = negative_slope
            self.dropout = dropout

        if bias:
            self.bias = Parameter(torch.Tensor(self.heads * out_channels if concat else out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        if self.bias is not None:
            zeros(self.bias)

    def message(self, x_j, norm, alpha=None):
        """
        定义消息传递函数
        Args:
            x_j: [E, heads * out_channels] 或 [E, heads, out_channels]
            norm: [E,] 归一化系数
            alpha: [E, heads, 1] 注意力权重 (可选)
        Returns:
            [E, heads, out_channels] 消息
        """

        if self.use_attention:
            # 处理多头注意力情况
            x_j = x_j.view(-1, self.heads, self.out_channels)  # [E, heads, out]
            # print(x_j.shape)
            # print(norm.shape)
            msg = x_j * norm.view(-1, 1, 1)  # 应用归一化
            if alpha is not None:
                msg = msg * alpha  # 应用注意力权重
            # print(msg.shape) torch.Size([5253, 8, 64])
            return msg
        else:
            # 无注意力情况
            # print((x_j * norm.view(-1, 1) ).shape)
            return x_j * norm.view(-1, 1)  # [E, out]

    def aggregate(self, inputs, index, nodes=None,ptr=None, dim_size=None):
        # 显式定义聚合方式
        # print(index.shape)
        # print(dim_size)
        if nodes is not None:
            dim_size = nodes
        if self.use_attention:
            # 多头注意力的聚合
            # print("目标节点唯一值数量:", index.unique().size(0))
            return scatter(inputs, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='mean')

        return super().aggregate(inputs, index, ptr, dim_size)

    def forward(self, x, adj_matrix, hyperedge_weight=None, EW_weight=None):
        batch_size, num_nodes, _ = x.shape
        num_nodes, num_edges = adj_matrix.shape
        # 转换邻接矩阵为edge_index
        src_nodes, hyperedges = torch.nonzero(adj_matrix, as_tuple=True)
        hyperedge_index = torch.stack([src_nodes, hyperedges], dim=0)
        # 线性变换
        x_ = torch.matmul(x, self.weight)  # [batch, N, heads*out]

        # 注意力计算
        alpha = None
        if self.use_attention:
            alpha_list = []
            for b in range(batch_size):
                x_h = x_[b].view(-1, self.heads, self.out_channels)
                x_i = x_h[hyperedge_index[0]]  # [E, heads, out]
                x_j = x_h[hyperedge_index[1]]  # [E, heads, out]

                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
                alpha = F.leaky_relu(alpha, self.negative_slope)
                alpha = softmax(alpha, hyperedge_index[0])
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
                alpha = alpha.view(-1, self.heads, 1)  # [E, heads, 1]
                alpha_list.append(alpha)
            alpha = torch.stack(alpha_list)

        # 准备传播参数
        # x_flat = x.view(batch_size * num_nodes, -1)
        D_list = []
        for b in range(batch_size):
            D = scatter_add(hyperedge_weight[b][hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
            D = 1.0 / D  # all 0.5 if hyperedge_weight is None
            D[D == float("inf")] = 0
            D_list.append(D)
        D = torch.stack(D_list)

        B_list = []
        for b in range(batch_size):
            if EW_weight is None:
                B = scatter_add(x[b].new_ones(hyperedge_index.size(1)),
                                hyperedge_index[1], dim=0, dim_size=num_edges)
            else:
                B = scatter_add(EW_weight[b][hyperedge_index[0]],
                    hyperedge_index[1], dim_size=num_edges)
            B = 1.0 / B
            B[B == float("inf")] = 0
            B_list.append(B)
        B = torch.stack(B_list)
        out_list = []
        for b in range(batch_size):
            # 第一轮传播 (节点->超边)
            if alpha is not None:
                alpha_ = alpha[b]
            else:
                alpha_ = None
            self.flow = 'source_to_target'
            # print(edge_index.shape)
            # print(edge_index.max().item())
            # print(edge_index[1].max().item())
            # print(edge_index[0].unique().size(0)) 153
            # print(edge_index[1].unique().size(0)) 51
            # print(num_nodes)
            # print(num_edges)
            out = self.propagate(
                hyperedge_index,
                x=x_[b],
                norm=B[b][hyperedge_index[1]],  # 使用B归一化
                alpha=alpha_,
                nodes=num_nodes,
                size=(num_nodes, num_edges))
            # 第二轮传播 (超边->节点)
            self.flow = 'target_to_source'
            out = self.propagate(
                hyperedge_index,
                x=out,
                norm=D[b][hyperedge_index[0]],  # 使用D归一化
                alpha=alpha_,
                nodes=num_nodes,
                size=(num_edges, num_nodes))
            out_list.append(out)

        # 处理输出
        out = torch.stack(out_list)
        out = out.contiguous().view(batch_size, num_nodes, -1)
        if not self.concat or self.heads == 1:
            out = out.mean(dim=-1) if self.heads > 1 else out.squeeze(-1)
        if self.bias is not None:
            out = out + self.bias
        out = self.dense(out)
        out = self.act(out) + x

        return out

class HyperGCN(nn.Module):
    def __init__(self, config, in_channels, out_channels, layers, heads, use_attention=True,
                 concat=True, negative_slope=0.2, dropout=0.3, bias=True):
        super(HyperGCN, self).__init__()
        self.lens = 51

        self.layer_num = layers
        self.layers = nn.ModuleList([])
        self.dense = nn.ModuleList([
            nn.Conv1d(self.lens*3, self.lens*2, kernel_size=1),
            nn.Conv1d(self.lens*2, self.lens, kernel_size=1)
        ])
        for i in range(self.layer_num):
            self.layers.append(nn.ModuleList([
                HypergraphConv(in_channels, out_channels, heads=heads, use_attention=use_attention,
                 concat=concat, negative_slope=negative_slope, dropout=dropout, bias=bias),
            ]))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()


    def forward(self, x, adj, hyperedge_weight=None, is_save=False):
        if is_save == True:
            out_list = []
            out_list.append(x)
            for i, layer in enumerate(self.layers):
                x = layer[0](x, adj[i], hyperedge_weight)
                if i <= 1:
                    x = self.dense[i](x)
                out_list.append(x)
            return out_list
        else:
            for i, layer in enumerate(self.layers):
                x = layer[0](x, adj[i], hyperedge_weight)
                if i <= 1:
                    x = self.dense[i](x)
            x = self.act(self.dropout(x))
            return x

