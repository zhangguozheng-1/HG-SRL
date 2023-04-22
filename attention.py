import numpy as np
import torch
from torch import optim
import math
# from metric import get_mrr, get_recall
import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pickle
from entmax import entmax_bisect


class DualAttention(nn.Module):
    def __init__(self, item_dim, pos_dim, n_items, w, atten_way='dot', decoder_way='bilinear', dropout=0,
                 activate='relu'):
        super(DualAttention, self).__init__()
        self.item_dim = item_dim
        self.pos_dim = pos_dim
        dim = item_dim
        self.dim = item_dim
        self.n_items = n_items
        # self.embedding = nn.Embedding(n_items + 1, item_dim, padding_idx=0, max_norm=1.5)
        # self.pos_embedding = nn.Embedding(n_pos, pos_dim, padding_idx=0, max_norm=1.5)
        self.atten_way = atten_way
        self.decoder_way = decoder_way
        self.atten_w0 = nn.Parameter(torch.Tensor(1, dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_bias = nn.Parameter(torch.Tensor(dim))
        self.dropout = nn.Dropout(dropout)
        self.self_atten_w1 = nn.Linear(dim, dim)
        self.self_atten_w2 = nn.Linear(dim, dim)

        self.LN = nn.LayerNorm(dim)
        self.LN2 = nn.LayerNorm(item_dim)
        self.is_dropout = True
        self.attention_mlp = nn.Linear(dim, dim)
        self.alpha_w = nn.Linear(dim, 1)
        self.w = w

        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'selu':
            self.activate = F.selu

        self.initial_()

    def initial_(self):
        init.normal_(self.atten_w0, 0, 0.05)
        init.normal_(self.atten_w1, 0, 0.05)
        init.normal_(self.atten_w2, 0, 0.05)
        init.constant_(self.atten_bias, 0)
        init.constant_(self.attention_mlp.bias, 0)
        # init.constant_(self.embedding.weight[0], 0)
        # init.constant_(self.pos_embedding.weight[0], 0)

    def forward(self, x_embeddings, pos_embeddings, mask):
        self.is_dropout = True
        mask = torch.cat([mask, mask], dim=-1)
        # print(item_embedding.shape)     # 【40841，100】，项目的嵌入比项目数量多1，是因为在这个嵌入的第一维填了一行0
        # print(x_embeddings.shape)       # 【B，seq_len，emb_size】
        # print(pos_embeddings.shape)     # 【B，seq_len，emb_size】
        x_ = torch.cat((pos_embeddings, x_embeddings), 1)  # B 2*seq, dim,拼接会话嵌入和邻居会话项目嵌入
        x_s = x_[:, :-1, :]  # B, 2*seq-1, dim
        # print("x_s.shape = ", x_s.shape)  #
        alpha_ent = self.get_alpha(x=x_embeddings[:, -1, :], number=0, len=x_.shape[1])
        # print("alpha_ent.shape = ", alpha_ent.shape)
        m_s, x_n = self.self_attention(x_, x_, x_, mask, alpha_ent)
        # print("m_s.shape = ", m_s.shape)    # m_s.shape =  torch.Size([B, 1, dim]),最后一个项目
        # print("x_n.shape = ", x_n.shape)    # x_n.shape =  torch.Size([B, seq-1, dim])，除去最后一个项目
        local_c = torch.cat([m_s, x_n], dim=1)
        # print(local_c.shape)
        alpha_global = self.get_alpha(x=m_s, number=1)
        # print("alpha_global.shape = ", alpha_global.shape)  # alpha_global.shape =  torch.Size([B, 1, 1])
        global_c = self.global_attention(m_s, x_n, x_s, mask, alpha_global)  # B, 1, dim
        # print("global_c.shape = ", global_c.shape)    # h_t.shape =  torch.Size([B, 1, dim])
        # print("local_c.shape = ", local_c.shape)    # m_s.shape =  torch.Size([B, 1, dim])
        return global_c, local_c
        # result = self.decoder(item_embedding, h_t, m_s, A)
        # return result

    def get_alpha(self, x=None, number=None, len=None):
        if number == 0:
            alpha_ent = torch.sigmoid(self.alpha_w(x)) + 1
            alpha_ent = self.add_value(alpha_ent).unsqueeze(1)
            # print("alpha_ent = ", alpha_ent.shape)
            alpha_ent = alpha_ent.expand(-1, len, -1)
            return alpha_ent
        if number == 1:
            alpha_global = torch.sigmoid(self.alpha_w(x)) + 1
            alpha_global = self.add_value(alpha_global)
            return alpha_global

    def add_value(self, value):

        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value

    def self_attention(self, q, k, v, mask=None, alpha_ent=1):
        # print("q.shape ", q.shape)
        # print("k.shape ", k.shape)
        # print("v.shape ", v.shape)
        # print("mask.shape = ", mask.shape)
        if self.is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))
        else:
            q_ = self.activate(self.attention_mlp(q))
        scores = torch.matmul(q_, k.transpose(1, 2)) / math.sqrt(self.dim)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
            scores = scores.masked_fill(mask == 0, -np.inf)
        # print("scores.shape", scores.shape)     # torch.Size([100, 18, 18])
        alpha = entmax_bisect(scores, alpha_ent, dim=-1)
        # print("alpha.shape = ", alpha.shape)
        att_v = torch.matmul(alpha, v)  # B, seq, dim   # 就是自注意力的输出 \hat A
        if self.is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v
        att_v = self.LN(att_v)
        # print("att_v.shape = ", att_v.shape)    # att_v.shape =  torch.Size([B, seq, 2 * dim])
        c = att_v[:, -1, :].unsqueeze(1)
        x_n = att_v[:, :-1, :]
        # output
        # c  ：c是学习到的目标嵌入，包含特殊的项目索引，并融合整个会话的信息来表示用户真实的当前偏好
        # x_n：这个经过自注意力和 前向反馈网络得到的 会话的新的项目嵌入
        return c, x_n

    def global_attention(self, target, k, v, mask=None, alpha_ent=1):
        alpha = torch.matmul(
            torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias),
            self.atten_w0.t())  # (B,seq,1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask = mask[:, :-1, :]
            alpha = alpha.masked_fill(mask == 0, -np.inf)
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        # v就是初始项目嵌入x_s，alpha就是论文中的β，就是当前会话st的注意力评分向量
        c = torch.matmul(alpha.transpose(1, 2), v)  # (B, 1, dim)
        return c

    # def decoder(self, item, global_c, self_c):
    #     if self.is_dropout:
    #         c = self.dropout(torch.selu(self.w_f_rec(torch.cat((global_c, self_c), 2))))
    #     else:
    #         c = torch.selu(self.w_f(torch.cat((global_c, self_c), 2)))
    #     c = c.squeeze()
    #     l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
    #     l_emb = item[1:] / torch.norm(item[1:], dim=-1).unsqueeze(1)
    #     z = self.w * torch.matmul(l_c, l_emb.t())
    #     return z

    # def predict(self, x, pos, k=20):
    #     self.is_dropout = False
    #     x_embeddings = self.embedding(x)  # B,seq,dim
    #     pos_embeddings = self.pos_embedding(pos)  # B, seq, dim
    #     mask = (x != 0).float()  # B,seq
    #     x_ = torch.cat((x_embeddings, pos_embeddings), 2)  # B seq, 2*dim
    #     x_s = x_[:, :-1, :]  # B, seq-1, 2*dim
    #     alpha_ent = self.get_alpha(x=x_[:, -1, :], number=0)
    #     m_s, x_n = self.self_attention(x_, x_, x_, mask, alpha_ent)
    #     alpha_global = self.get_alpha(x=m_s, number=1)
    #     global_c = self.global_attention(m_s, x_n, x_s, mask, alpha_global)  # B, 1, dim
    #     h_t = global_c
    #     result = self.decoder(h_t, m_s)
    #     rank = torch.argsort(result, dim=1, descending=True)
    #     return rank[:, 0:k]


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general2'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            torch.nn.init.normal_(self.transform.weight, std=0.01)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """

        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            # print(M.shape)
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)  # batch, seq_len, mem_dim
            # print("mask.shape = ", mask.shape)
            # print("mask111.shape = ", mask_.shape)
            # print(M_.shape)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
            # import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim
        return attn_pool, alpha