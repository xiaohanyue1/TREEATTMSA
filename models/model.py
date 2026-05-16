import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from models.att_gnn import TransformerEncoder_ver1, Dispersed_Graph, TransformerEncoder_ver2, TransformerEncoder
from models.htygraph import HyperGCN

class Positional_Embedding(nn.Module):
    def __init__(self, config, embed_size):
        super(Positional_Embedding, self).__init__()
        self.dropout = nn.Dropout(config.p_drop)
        pe = torch.zeros(51, embed_size)  # 只创建了50的
        position = torch.arange(0, 51, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # print(self.pe[:x.size(0), :].shape)
        # print(x.size(0))
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0), :]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class LGMSAT(nn.Module):
    def __init__(self, config):
        super(LGMSAT, self).__init__()
        self.device = 'cuda'
        # bert
        if config['datasetName'] == 'sims':
            self.bert = BertModel.from_pretrained(config.ch_pretrained_path)
        else:
            self.bert = BertModel.from_pretrained(config.en_pretrained_path)
        # cls
        self.t_cls = nn.Parameter(torch.randn((1, 1, config['t_size_in']), device=self.device))
        nn.init.normal_(self.t_cls)
        self.v_cls = nn.Parameter(torch.randn((1, 1, config['v_size_in']), device=self.device))
        nn.init.normal_(self.v_cls)
        self.a_cls = nn.Parameter(torch.randn((1, 1, config['a_size_in']), device=self.device))
        nn.init.normal_(self.a_cls)
        # position
        self.gru_t = nn.GRU(input_size=config['t_size_in'], hidden_size=64, num_layers=2,
                   batch_first=True, bidirectional=True)
        self.po_t = Positional_Embedding(config, config['t_in_dim'])

        self.gru_v = nn.GRU(input_size=config['v_size_in'], hidden_size=64, num_layers=2,
                   batch_first=True, bidirectional=True)
        self.po_v = Positional_Embedding(config, config['v_in_dim'])

        self.gru_a = nn.GRU(input_size=config['a_size_in'], hidden_size=64, num_layers=2,
                   batch_first=True, bidirectional=True)
        self.po_a = Positional_Embedding(config, config['a_in_dim'])
        # # embedding
        self.t_emb = TransformerEncoder_ver1(in_dim=config['V_dim'], out_dim=config['V_out_dim'], depth=config['V_depth'], 
                                            heads=config['V_heads'], dim_head=64, mlp_dim=config['V_mlp_dim'], dropout=0.3)
        self.v_emb = TransformerEncoder_ver1(in_dim=config['V_dim'], out_dim=config['V_out_dim'], depth=config['V_depth'],
                                            heads=config['V_heads'], dim_head=64, mlp_dim=config['V_mlp_dim'], dropout=0.3)
        self.a_emb = TransformerEncoder_ver1(in_dim=config['V_dim'], out_dim=config['V_out_dim'], depth=config['V_depth'],
                                            heads=config['V_heads'], dim_head=64, mlp_dim=config['V_mlp_dim'], dropout=0.3)
        # nonverbal reconstruction
        self.t_re = TransformerEncoder_ver2(in_dim=config['V_dim'], out_dim=config['V_out_dim'], depth=config['V_depth'],
                                           heads=config['V_heads'], dim_head=64, mlp_dim=config['V_mlp_dim'], dropout=0.1)
        self.v_re = TransformerEncoder_ver2(in_dim=config['V_dim'], out_dim=config['V_out_dim'], depth=config['V_depth'],
                                           heads=config['V_heads'], dim_head=64, mlp_dim=config['V_mlp_dim'], dropout=0.1)
        self.a_re = TransformerEncoder_ver2(in_dim=config['V_dim'], out_dim=config['V_out_dim'], depth=config['V_depth'],
                                           heads=config['V_heads'], dim_head=64, mlp_dim=config['V_mlp_dim'], dropout=0.1)

        # recall
        self.v_rc = TransformerEncoder_ver2(in_dim=config['rc_dim'], out_dim=config['rc_out_dim'], depth=config['rc_depth'],
                                           heads=config['rc_heads'], dim_head=64, mlp_dim=config['rc_mlp_dim'], dropout=0.5)
        self.a_rc = TransformerEncoder_ver2(in_dim=config['rc_dim'], out_dim=config['rc_out_dim'], depth=config['rc_depth'],
                                           heads=config['rc_heads'], dim_head=64, mlp_dim=config['rc_mlp_dim'], dropout=0.5)

        # decoder
        self.dispersed_graph_v = Dispersed_Graph(depth=config['dis_depth'], in_features=config['dis_in_feature'], dropout=0.5, 
                                                 alpha=0.2, hidden_size=config['dis_hidden_size'], share_weight=False, concat=False)
        self.dispersed_graph_a = Dispersed_Graph(depth=config['dis_depth'], in_features=config['dis_in_feature'], dropout=0.5, 
                                                 alpha=0.2, hidden_size=config['dis_hidden_size'], share_weight=False, concat=False)

        self.linear = nn.Linear(128 * 2, 128)
        self.act = nn.ReLU()
        self.dense = nn.Linear(128,  1)


    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std * eps

    def creat_adj(self, nodes):
        adj_list = []
        # 对角
        adj1 = torch.diag(torch.ones(nodes)).cuda()
        # 上三角
        adj2 = torch.triu(torch.ones(nodes, nodes)).cuda()
        # 下三角
        adj3 = torch.tril(torch.ones(nodes, nodes)).cuda()
        adj_list.append(adj1)
        adj_list.append(adj2)
        adj_list.append(adj3)

        return adj_list


    def forward(self, text, visual, audio, text_mask):
        batch_size, nodes = text.shape
        # or_t
        t_bert = self.bert(text, text_mask)[0]
        t_cls = self.t_cls.repeat(batch_size, 1, 1)
        or_t = torch.cat((t_cls, t_bert), dim=1)
        or_t, _ = self.gru_t(or_t)
        or_t = self.po_t(or_t)
        t_emb_list = self.t_emb(or_t,save_hidden=True)
        # or_v
        v_cls = self.v_cls.repeat(batch_size, 1, 1)
        or_v = torch.cat((v_cls, visual), dim=1)
        or_v, _ = self.gru_v(or_v)
        or_v = self.po_v(or_v)
        v_emb_list = self.v_emb(or_v, save_hidden=True)
        # or_a
        a_cls = self.a_cls.repeat(batch_size, 1, 1)
        or_a = torch.cat((a_cls, audio), dim=1)
        or_a, _ = self.gru_a(or_a)
        or_a = self.po_a(or_a)
        a_emb_list = self.a_emb(or_a, save_hidden=True)
        # nonverbal reconstruction
        t_re_list = self.t_re(t_emb_list[0], t_emb_list, save_hidden=True)
        v_re_list = self.v_re(t_emb_list[0], v_emb_list, save_hidden=True)
        a_re_list = self.a_re(t_emb_list[0], a_emb_list, save_hidden=True)
        # adj
        adj_list = self.creat_adj(nodes+1)
        # gnn decoder
        # print(v_re_list[-1].shape)
        vt_graph = self.dispersed_graph_v(v_re_list[-1], adj_list)
        at_graph = self.dispersed_graph_a(a_re_list[-1], adj_list)
        # recall
        t_re_list_rv = t_re_list[::-1]
        v_rc = self.v_rc(vt_graph, t_re_list_rv, save_hidden=False)
        a_rc = self.a_rc(at_graph, t_re_list_rv, save_hidden=False)
        # output
        output = torch.cat((v_rc, a_rc), dim=-1)
        output = self.linear(output)
        output = self.act(output)
        output = self.dense(output)[:,0]
        # mu = self.mu(output)
        # logvar = self.logvar(output)
        # z = self.reparameterise(mu, torch.exp(logvar))
        # out = self.out(mu)


        # return out, z, mu, logvar, t_emb, v_emb, a_emb
        return output

