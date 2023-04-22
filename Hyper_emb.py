import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from numba import jit
from util import get_data, get_overlap
import random
import operator
from attention import DualAttention, MatchingAttention
# from common.memory import ReplayBuffer
from functools import partial
from collections import Counter


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda(0)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = item_embedding_layer0
        # final1 = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            # print(item_embeddings.shape)
            # print(item_embeddings.device)
            final = final + item_embeddings
            # final1.append(item_embeddings)
        # final111 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final1]))
        # item_embeddings = torch.sum(final111, 0)
        # print("final = ", final)
        # print("item_embedding = ", item_embeddings)
        # item_embeddings = np.sum(final, 0)
        item_embeddings = final
        return item_embeddings
    # def forward(self, adjacency, embedding):
    #     item_embeddings = embedding
    #     item_embedding_layer0 = item_embeddings
    #     final = [item_embedding_layer0]
    #     final = torch.zeros([self.layers+1, 1])
    #     for i in range(self.layers):
    #         item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
    #         # print(item_embeddings.shape)
    #         # print(item_embeddings.device)
    #         final.append(item_embeddings)
    #     # final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
    #     # item_embeddings = torch.sum(final1, 0)
    #     item_embeddings = np.sum(final, 0)
    #     return item_embeddings


class LineConv(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers

    def forward(self, item_embedding, D, A, session_item, session_len):
        # zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        zeros = trans_to_cuda(torch.zeros([1,self.emb_size]))
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)

        session = [session_emb_lgcn]

        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)

        # session1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        # session_emb_lgcn = torch.sum(session1, 0)

        session_emb_lgcn = np.sum(session, 0)
        return session_emb_lgcn


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, A, hidden, alias_pos):
        """其中邻接矩阵是根据顺序点击顺序构造的，门控网络用于动态控制应分别地从相邻节点和星形节点获取多少信息"""
        # print(A.shape)
        # print(torch.tensor(A[:, :, :A.shape[1]]).shape)   # torch.Size([B, max_node , max_node])
        # print(hidden.shape)    # torch.Size([B, max_node , emb_size ])
        # get = lambda i: hidden[i][alias_pos[i]]
        # seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_pos)).long()])
        # print("AAA:", A)
        # alias_pos = torch.tensor([[0, 2, 2, -1, -1]]).long()
        # print("pos : ", alias_pos)
        # get = lambda i: A[i][alias_pos[i]]
        get = lambda i: A[i][alias_pos[i]]
        # in_out_adj = torch.empty_like(A)
        in_out_adj = torch.stack([get(i) for i in torch.arange(len(alias_pos)).long()])
        # in_out_adj = [get(i) for i in range(len(alias_pos))]
        # trans_to_cuda(in_out_adj.append([get(i) for i in range(len(alias_pos))]))
        # print(A)
        # print("linjie :", in_out_adj)
        # in_out_adj = A
        input_in = torch.matmul(in_out_adj[:, :, :in_out_adj.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(in_out_adj[:, :, in_out_adj.shape[1]: 2 * in_out_adj.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy


class DHCN(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, beta, dataset, emb_size=100, batch_size=100):
        super(DHCN, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta

        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        if dataset == 'Nowplaying':
            index_fliter = (values < 0.05).nonzero()
            values = np.delete(values, index_fliter)
            indices1 = np.delete(indices[0], index_fliter)
            indices2 = np.delete(indices[1], index_fliter)
            indices = [indices1, indices2]
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        self.adj_index = i
        self.adj_value = v
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.adjacency = adjacency
        # print(self.adjacency)
        self.neighor_item = trans_to_cuda(self.gen_neighor_item())
        # print(self.neighor_item.shape)
        # print(self.neighor_item[:20])
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_embedding = nn.Embedding(200, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset)
        self.LineGraph = LineConv(self.layers, self.batch_size)
        self.GraphNN = GNN(self.emb_size)

        self.dual_att = DualAttention(emb_size, emb_size, n_node, 20, dropout=0.5, activate='relu')
        self.match_att = MatchingAttention(self.emb_size, self.emb_size, att_type='general2')

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

        self.linear_hn = nn.Linear(2 * self.emb_size, self.emb_size)
        self.w_f_rec = nn.Linear(4 * self.emb_size, self.emb_size)
        self.w_f_rl = nn.Linear(4 * self.emb_size, self.emb_size)
        self.rec_lin = nn.Linear(self.emb_size, self.emb_size)
        self.fc = nn.Linear(2 * self.emb_size, self.n_node)
        self.dropout = nn.Dropout(0.2)
        self.w = 20
        # self.enc_mask_token = nn.Parameter(torch.zeros(1, self.emb_size))
        # self.decoder = nn.Sequential(nn.Linear(self.emb_size, self.emb_size),
        #                                 nn.PReLU(),
        #                                 nn.Dropout(0.2),
        #                                 nn.Linear(self.emb_size, self.emb_size) )
        # self.encoder_to_decoder = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.criterion = partial(self.sce_loss, alpha=3)
        # self.margin_loss = nn.MarginRankingLoss(margin=0.8, reduce=False)
        # self.full_connect_weight = nn.Linear(self.emb_size, self.n_node)

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sce_loss(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

    def decoder1(self, item, state):  # 推荐的MLP层
        # c = self.dropout(torch.selu(self.w_f_rec(torch.cat((global_c, self_c), 2))))
        c = torch.selu(state)
        c = c.squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = item[0:] / torch.norm(item[0:], dim=-1).unsqueeze(1)
        z = self.w * torch.matmul(l_c, l_emb.t())
        return z

    def decoder2(self, item, state):  # RL的MLP层
        # c = self.dropout(torch.selu(self.w_f_rec(torch.cat((global_c, self_c), 2))))
        c = torch.selu(state)
        c = c.squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = item[0:] / torch.norm(item[0:], dim=-1).unsqueeze(1)
        z = self.w * torch.matmul(l_c, l_emb.t())
        return z

    def decoder(self, item, state):  # 推荐的MLP层
        # c = self.dropout(torch.selu(self.w_f_rec(torch.cat((global_c, self_c), 2))))
        # c = torch.selu(state)
        c = self.dropout(torch.selu(self.rec_lin(state)))
        # print(c.shape)
        c = c.squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = item[0:] / torch.norm(item[0:], dim=-1).unsqueeze(1)
        # z = self.fc(c)
        z = self.w * torch.matmul(l_c, l_emb.t())
        return z

    def Matching(self, matchatt, emotions, modal, umask):
        att_emotions = []
        alpha = []
        for t in modal:
            att_em, alpha_ = matchatt(emotions, t, mask=umask)
            # print(att_em.shape)
            att_emotions.append(att_em.unsqueeze(0))
            alpha.append(alpha_[:, 0, :])
        att_emotions = torch.cat(att_emotions, dim=0)
        hidden = att_emotions + F.gelu(emotions)
        return hidden, alpha

    def gen_neighor_item(self):
        # origin_index = torch.where(self.adj_value > 0)[0]
        # print(origin_index)
        # print(origin_index.size())  # 4w多的一个数
        drop = torch.bernoulli(self.adj_value)
        index = torch.where(drop > 0)[0]
        # print(index)
        # print(index.size())  # 4w多的一个数
        # print(self.adj_index.size())  # 2*关系数量1933303
        item_index_i = []
        item_index_ij = []
        for j in index:
            item_index_i.append(self.adj_index[0][j].tolist())
            item_index_ij.append(self.adj_index[1][j].tolist())
        # print(item_index_i)
        # print(item_index_ij)
        i_times = Counter(item_index_i)
        ij_times = Counter(item_index_ij)
        # print(i_times)
        # print(ij_times)
        max_i_times = i_times.most_common(1)
        max_ij_times = ij_times.most_common(1)
        # print(max_i_times)
        # print(max_ij_times)
        max_neighbor_num = max_i_times[0][1]
        # print(max_neighbor_num)
        neighbor_item = (np.zeros([self.n_node, max_neighbor_num]).astype(int) - 1)
        # print(neighbor_item.shape)
        # next_item = 0
        i_neig_num = 0
        # print(len(item_index_i) - 1)
        # print(item_index_i[:20])
        # print(item_index_ij[:20])

        # 找出每个项目邻接的项目，存在一个numpy类型的数组里，大小为【项目数量，每个项目的最大邻接项目数量】。
        # 每一行中前n个项目是其邻接的项目，其余的是填充的0
        for i in range(len(item_index_i) - 1):
            if i == 0:
                if item_index_i[0] >= 1:
                    cap = item_index_i[0] - 0
                    for xx in range(cap):
                        neighbor_item[xx][0] = xx

            if item_index_i[i] == item_index_i[i + 1]:
                i_neig_num += 1
            else:
                i_neig_num = 0
                cap = item_index_i[i + 1] - item_index_i[i]
                if cap > 1:
                    for xx in range(1, cap):
                        cap_item = item_index_i[i] + xx
                        neighbor_item[cap_item][i_neig_num] = cap_item

            neighbor_item[item_index_i[i]][i_neig_num] = item_index_ij[i]

        # print(neighbor_item[:30])
        # print(neighbor_item.shape)
        return torch.tensor(neighbor_item)

    def generate_sess_emb(self, item_embedding, session_item, neig_sess, session_len, reversed_sess_item, mask, sess_adj, alias_pos, A=None):
        # 根据超图得到的项目嵌入矩阵获取每个会话的项目级嵌入表示global_sess
        # 根据超图得到的项目嵌入矩阵获取每个会话的项目级嵌入表示，将这个表示送到GNN得到local_sess
        # 根据超图得到的项目嵌入矩阵获取当前batch中每个会话的邻居会话的嵌入，然后这个嵌入要怎么处理？

        zeros = trans_to_cuda(torch.zeros(1, self.emb_size))
        item_embedding = torch.cat([zeros, item_embedding], 0)  # 填充一行0，和项目的ID对应上

        len = session_item.shape[1]  # 因为每个会话都是填充成长度一样的会话，因此这个值是一个定值
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

#         seq_h1 = item_embedding[session_item]  # seq_h1是正序的会话项目嵌入表示
        seq_h = item_embedding[reversed_sess_item]  # # seq_h1是逆序的会话项目嵌入表示
        neig_seq_h = item_embedding[neig_sess]  # neig_seq_h是当前batch中每个会话的邻居会话的项目级嵌入表示

        """先利用GNN融合局部会话图的嵌入的信息，学习会话级项目嵌入"""
        loc_seq = self.GraphNN(sess_adj, seq_h, alias_pos)

        """ 利用dual注意力机制计算当前会话和其邻居会话的聚合表示"""
        global_att, local_att = self.dual_att(seq_h, neig_seq_h, mask)
        # print("global_att = ", global_att.shape)    # global_att =  torch.Size([500, 1, 100])
        # print("local_att = ", local_att.shape)    # local_att =  torch.Size([500, 38, 100])
        # print("loc_seq = ", loc_seq.shape)    # loc_seq =  torch.Size([500, 19, 100])

        """使用match Att，输入seq_h、loc_state、mask"""
        local_att = local_att.permute(1, 0, 2)
        loc_seq = torch.cat([loc_seq, loc_seq], dim=1).permute(1, 0, 2)
        mask = torch.cat([mask, mask], dim=-1)
        feature = [local_att, loc_seq]
        output = 0.
        for i in feature:
            for j in feature:
                hid, alpha = self.Matching(self.match_att, i, j, mask)
                output += hid
        output = output.permute(1, 0, 2)
        pos_emb = torch.cat([pos_emb, pos_emb], dim=1)
        # print("output.shape = ", output.shape)
        # print("pos_emb.shape = ", pos_emb.shape)

        hs = torch.div(torch.sum(output, 1), 2*session_len)
        mask_loc = mask.float().unsqueeze(-1)
        hs = hs.unsqueeze(-2).repeat(1, 2*len, 1)
        nh = torch.matmul(torch.cat([pos_emb, output], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask_loc
        state = torch.sum(beta * output, 1)   # 【B, emb_size】
        # print("loc_state shape is ", loc_state.shape)
        # print(global_att.shape)
        # state = torch.cat([global_att, local_att, loc_state.unsqueeze(1)], 2)
        # print(state.shape)
        global_state = global_att.squeeze(1)
        # print(global_state.shape)
        state = (global_state + state) / 2.0
        # print(state)
        return state

    def SSL(self, sess_emb_hgnn, sess_emb_drop):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

        shuffle_emb = row_column_shuffle(sess_emb_hgnn)
        # s_p = F.pairwise_distance(sess_emb_hgnn, sess_emb_drop)
        # s_n = F.pairwise_distance(sess_emb_hgnn, shuffle_emb)
        # # print(s_p.shape)
        # # print(s_p)
        # # print(s_n.shape)
        # # print(s_n)
        # margin_label = -1 * torch.ones_like(s_p)
        # lbl_z = trans_to_cuda(torch.tensor([0.]))
        # loss_mar = 0
        # dropout_margin_n = 0
        # loss_mar = (self.margin_loss(s_p, s_n, margin_label)).mean()
        # dropout_margin_n = torch.max((s_n - s_p.detach() - 1.3), lbl_z).sum()
        #
        # ssl_loss = loss_mar * 1 + dropout_margin_n * 1
        # print(ssl_loss)
        # pos = score(sess_emb_hgnn, sess_emb_drop)
        # neg1 = score(sess_emb_drop, row_column_shuffle(sess_emb_hgnn))

        # s_p = F.pairwise_distance()
        # one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        # con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        # con_loss = -torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1)))
        # print(con_loss)
        # print(con_loss.shape)
        ssl_loss = triplet_loss(sess_emb_hgnn, sess_emb_drop, shuffle_emb) + 1e-8
        # print(ssl_loss)
        return ssl_loss

    def SSL1(self, sess_emb_hgnn, sess_emb_drop):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        sess_emb_hgnn = self.rec_lin(sess_emb_hgnn)
        sess_emb_drop = self.rec_lin(sess_emb_drop)
        pos = score(sess_emb_hgnn, sess_emb_drop)
        neg1 = score(sess_emb_drop, row_column_shuffle(sess_emb_hgnn))
        # print(pos)
        # print(neg1)
        # one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        one = zeros = trans_to_cuda(torch.ones(neg1.shape[0]))
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        # con_loss = -torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1)))
        # print(con_loss)
        return con_loss

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        # G 是当前batch的数据，X 是根据超图卷积得到的项目嵌入
        replace_rate = 0.1
        mask_token_rate = 1 - replace_rate
        num_nodes = self.n_node  # 获取所有节点数目
        perm = torch.randperm(num_nodes, device=x.device)  # 获取一个 节点总数打乱 的数组，便于选择mask节点（就是itemembedding）
        # num_mask_nodes = int(mask_rate * num_nodes)
        # print(perm)
        # random masking！！！！！！！！ 在这里进行mask的操作
        num_mask_nodes = int(mask_rate * num_nodes)  # 设置要mask掉的节点数量：mask率 * 节点的总数 （mask率一般设置为0.5# ）
        # print(num_mask_nodes)
        mask_nodes = perm[: num_mask_nodes]  # mask 的节点列表
        keep_nodes = perm[num_mask_nodes:]  # mask 之后保留下的节点列表
        # print(mask_nodes)
        # print(keep_nodes)
        # 进行随机替换，就是在已经被选中要mask的节点中，再随机（设置一个概率replace_rate，一般是0.1）选择一些节点让其保持不变或者用另一个随机标记替换他。
        if replace_rate > 0:
            num_noise_nodes = int(replace_rate * num_mask_nodes)  # 定义决定要mask的项目中 要 随机mask 的节点
            # print(num_noise_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)  # 获取一个 打乱的mask节点数量 数组
            token_nodes = mask_nodes[
                perm_mask[: int(mask_token_rate * num_mask_nodes)]]  # _mask_token_rate = 1 - _replace_rate
            noise_nodes = mask_nodes[perm_mask[-int(replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            # print(noise_to_be_chosen)
            out_x = x.clone()  # 复制一下特征矩阵
            out_x[token_nodes] = 0.0  # 这是真正mask的节点
            out_x[noise_nodes] = x[noise_to_be_chosen]  # 这是给的噪声mask节点
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token  # 这个就是要训练的mask节点的特征参数
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    # 测试的时候用推荐的线性层，训练的时候是强化学习的线性层
    def forward(self, session_item, session_len, reversed_sess_item, mask, sess_adj, alias_pos, A=None, D=None):
        # with torch.no_grad:  # .detach()
        # source = self.embedding.weight
        item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)
        # print(session_item.shape) # 【B,seq_len】
        neig_item = trans_to_cuda(torch.zeros_like(session_item[0]))
        neig_sess = trans_to_cuda(torch.zeros_like(session_item))
        # print(self.neighor_item[:30])
        s_i = 0
        for sess in session_item:
            # print("session = ", sess)
            sess = sess[torch.where(sess > 0)]
            # print("除去填充的0的会话：", sess)
            i_i = 0
            for item in sess:
                it = torch.where(self.neighor_item[item-1] > 0)[0].shape[0]
                # print("非-1邻居数量为", it)
                if it == 0:
                    sample_item_index = 0
                else:
                    sample_item_index = torch.randint(0, it, [1])
                item = self.neighor_item[item-1][sample_item_index]
                neig_item[i_i] = item + 1
                i_i += 1
            # neig_sess.append(neig_item)
            neig_sess[s_i] = neig_item
            s_i += 1
        # print("neig: ", neig_sess)
        # print(item_embeddings_hg)
        # num_runs = 5
        # item_emb = []
        # item_emb.append(item_embeddings_hg)
        # for _ in range(num_runs):  # 多视图？
        #     drop = torch.bernoulli(torch.sub(1, self.adj_value))
        #     drop_value = torch.mul(self.adj_value, drop)
        #     adj = torch.sparse.FloatTensor(self.adj_index, drop_value, torch.Size(self.adjacency.shape))
        #     i_emb = self.HyperGraph(adj, self.embedding.weight)
        #     item_emb.append(i_emb)
        #     item_embeddings_dropout = torch.stack(item_emb, 0)
        # item_embeddings_dropout = torch.mean(item_embeddings_dropout, dim=0)
        # item_e_con = torch.cat([source, item_embeddings_hg], -1)
        # alpha = self.linear_hn(item_e_con)
        # alpha = torch.sigmoid(alpha)
        # item_embeddings_hg = alpha * source + (1 - alpha) * item_embeddings_hg

        # print(1 - self.adj_value)
        # print(torch.sub(1, self.adj_value))
        # 输入：项目嵌入、batch会话项目、每个项目的邻居项目、batch中每个会话的长度、
        # batch反转的会话项目、batch会话的mask、batch中每个会话局部图的邻接矩阵、标记每个会话中项目的位置
        state = self.generate_sess_emb\
            (item_embeddings_hg, session_item, neig_sess, session_len, reversed_sess_item, mask, sess_adj, alias_pos)
        # 设想：三个部分
        # 全局超图、会话局部图、会话中每个项目的邻居项目构成的邻居会话（这个需要构图吗？）？

        # state_drop = self.generate_sess_emb(item_embeddings_dropout, session_item, session_len, reversed_sess_item,
        #                                     mask, sess_adj, alias_pos)
        return item_embeddings_hg, state
        # recom_scores = self.decoder(item_embeddings_hg, state)
        # # print(recom_scores.shape)
        # # print(recom_scores.shape)
        # # print(recom_scores)
        # if A:
        #     ssl_loss = self.SSL(state, state_drop)
        #     return recom_scores, ssl_loss
        # #
        # # rec_state = self.w_f_rec(state)
        # # rl_state = self.w_f_rl(state)
        # # scores_rec = self.decoder1(item_embeddings_hg, rec_state)
        # # scores_rl = self.decoder2(item_embeddings_hg, rl_state)
        # # with open("log/scores_rec.txt", "a") as f:  # 记录每个epoch训练的输出，
        # #     f.write(str(scores_rec) + "\n")
        # # with open("log/scores_rl.txt", "a") as f:  # 记录每个epoch训练的输出，
        # #     f.write(str(scores_rl) + "\n")
        # # print("rec*---", self.w_f_rec.weight)
        # # print("rl*----", self.w_f_rl.weight)
        # # if A:
        # #     # print(self.adjacency)
        # #     # print(type(self.adjacency))
        # #     num_runs = 5
        # #     item_emb = []
        # #     for _ in range(num_runs):   # 多视图？
        # #         drop = torch.bernoulli(torch.sub(1, self.adj_value))
        # #         drop_value = torch.mul(self.adj_value, drop)
        # #         adj = torch.sparse.FloatTensor(self.adj_index, drop_value, torch.Size(self.adjacency.shape))
        # #         i_emb = self.HyperGraph(adj, self.embedding.weight)
        # #         item_emb.append(i_emb)
        # #         item_embeddings_dropout = torch.stack(item_emb, 0)
        # #     item_embeddings_dropout = torch.mean(item_embeddings_dropout, dim=0)
        # #     state_drop = self.generate_sess_emb(item_embeddings_dropout, session_item, session_len, reversed_sess_item, mask)
        # #     rec_drop_state = self.w_f_rec(state_drop)
        # #     con_loss = self.SSL1(rec_state, rec_drop_state, rl_state)
        # #     # con_loss = self.SSL(state, state_drop)
        # #     # scores_rec = self.decoder1(item_embeddings_hg, state)
        # #     # scores_rl = self.decoder2(item_embeddings_hg, state)
        # #     # print(drop)
        # #     # self.adjacency[drop] = torch.zeros([drop.sum().long().item(), self.adjacency.size(-1)])
        # #     # del drop
        # #
        # #     # batch_data, item_emb_weight, (mask_nodes, keep_nodes) = self.encoding_mask_noise(session_item,
        # #
        # #     # x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
        # #     # drop  = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * p).bool()
        # #     # x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
        # #     # del drop
        # #
        # #
        # #     # batch_data, item_emb_weight, (mask_nodes, keep_nodes) = self.encoding_mask_noise(session_item,
        # #     #                                                                                  self.embedding.weight)
        # #     # # 超图编码encode部分，使用的是超图卷积，
        # #     # mask_item_emb = self.HyperGraph(self.adjacency, item_emb_weight)
        # #     # rep = self.encoder_to_decoder(mask_item_emb)
        # #     # # re-mask操作
        # #     # rep[mask_nodes] = 0.
        # #     #
        # #     # score1 = self.generate_sess_emb(rep, session_item, session_len, reversed_sess_item, mask)
        # #     # mask_loss = self.criterion(score, score1)
        # #     return scores_rec, scores_rl, con_loss
        # # recon = self.HyperGraph(self.adjacency, rep)  # 使用超图进行解码，得到新的mask后的项目嵌入向量
        # # recon = self.decoder(rep)   # 使用MLP进行解码
        # # print(recon.shape)    # 【43097，100】
        # # x_init = mask_item_emb[mask_nodes]
        # # x_rec = recon[mask_nodes]
        # # mask_loss = self.criterion(x_rec, x_init)
        #
        # # item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)
        # # sess_emb_hgnn就是模型得到的用户的选择结果select
        # # sess_emb_hgnn = self.generate_sess_emb(self.embedding.weight, session_item, session_len, reversed_sess_item, mask)
        #
        # # score1 = self.generate_sess_emb(item_embeddings_hg, session_item, session_len, reversed_sess_item, mask)
        # # mask_loss = self.criterion(x_rec, x_init)
        # return recom_scores

