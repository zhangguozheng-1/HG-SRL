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
from attention import DualAttention
# from common.memory import ReplayBuffer
from functools import partial
from Hyper_emb import DHCN

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

class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a #两个超参数
        self.b = b
        self.eps = 1e-7
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CE 部分，正常的交叉熵损失
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0) #最小设为 1e-4，即 A 取 -4
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss

class Qnetwork(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, beta, dataset, reward_click, reward_buy, discount, state_size, emb_size=100, batch_size=100):
        super(Qnetwork, self).__init__()
        self.emb_size = emb_size    # 状态的嵌入维度
        self.batch_size = batch_size
        self.n_node = n_node    # 这个表示的是项目的数量
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.dataset = dataset
        self.adjacency = adjacency
        self.reward_click = reward_click
        self.reward_buy = reward_buy
        self.discount = discount
        self.state_size = state_size

        # e-greedy策略相关参数
        self.actions_count = 0
        self.epsilon_start = 1  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.epsilon = 0
        # Q网络计算loss用的折损系数
        self.gamma = 0.9

        self.lr = lr
        self.DHCNrec = DHCN(self.adjacency, self.n_node, self.lr, self.layers, self.L2,
                            self.beta, self.dataset, self.emb_size, self.batch_size)
        self.loss = 0
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.2)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
#         self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = SCELoss(43097, 0.8, 0.5)
        
        # self.w_f_rec = nn.Linear(4 * self.emb_size, self.emb_size)
        # self.w_f_rl = nn.Linear(4 * self.emb_size, self.emb_size)放开心
        self.rec_lin = nn.Linear(self.emb_size, self.emb_size)
        self.rec_lin1 = nn.Linear(self.emb_size, self.emb_size)
        self.w_f_rec = nn.Linear(4 * self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(0.5)
        self.dropout1 = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * self.emb_size, self.n_node)
        self.w = 20
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

        # self.full_connect_weight = nn.Linear(self.emb_size, self.n_node)
        # output1 = self.full_connect_weight(hidden_state)    # output1推荐模型得到的选择项目的概率Q—value
        # output2 = self.full_conn ect_weight(hidden_state)    # 这里其实也可以换成计算scores那个代码

    def decoder(self, item, state, mode='train'):  # 推荐的MLP层
        if mode=="train":
            c = self.dropout(torch.selu(self.rec_lin(state)))
        else:
            c = torch.selu(self.rec_lin(state))
        # print(c.shape)
        c = c.squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = item[0:] / torch.norm(item[0:], dim=-1).unsqueeze(1)
        # z = self.fc(c)
        z = self.w * torch.matmul(l_c, l_emb.t())
        return z

    def decoder1(self, item, state):  # 推荐的MLP层
        # c = self.dropout(torch.selu(self.w_f_rec(torch.cat((global_c, self_c), 2))))
        # c = torch.selu(state)
        # c = torch.selu(self.rec_lin1(state))
        c = self.dropout1(torch.selu(self.rec_lin1(state)))

        # print(c.shape)
        c = c.squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = item[0:] / torch.norm(item[0:], dim=-1).unsqueeze(1)
        # z = self.fc(c)
        z = self.w * torch.matmul(l_c, l_emb.t())
        return z

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
        ssl_loss = self.triplet_loss(sess_emb_hgnn, sess_emb_drop, shuffle_emb) + 1e-8
        # print(ssl_loss)
        return ssl_loss

    def gen_rec(self, state, len_state, A_hat=None, D_hat=None):
        session_item = trans_to_cuda(torch.Tensor(state[0]).long())
        reversed_sess_item = trans_to_cuda(torch.Tensor(state[1]).long())
        mask = trans_to_cuda(torch.Tensor(state[2]).long())
        sess_adj = trans_to_cuda(torch.Tensor(state[3]).float())
        alias_pos = trans_to_cuda(torch.Tensor(state[4]).long())
        # session_len = trans_to_cuda(torch.Tensor(len_state).long())

        session_len = trans_to_cuda(torch.reshape(torch.Tensor(len_state).long(), [self.batch_size, 1]))
        # reshape成一个二维的数组，因为超图计算中要用这样的形式[[session_len1],[session_len2], ...[],[]]
        # print(session_item)
        item_embeddings_hg, state \
            = self.DHCNrec(session_item, session_len, reversed_sess_item, mask, sess_adj, alias_pos, A_hat)

        if A_hat:
            recom_scores = self.decoder(item_embeddings_hg, state)
            q_scores = self.decoder1(item_embeddings_hg, state)
            return q_scores, recom_scores
        else:
            recom_scores = self.decoder(item_embeddings_hg, state, mode='test')
            return recom_scores

    def choice_action(self, prob_Q):
        self.actions_count += 1     # 这个值会一直增加
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.actions_count / self.epsilon_decay)

        if random.random() > self.epsilon:    # 改成这个就是贪婪策略
        # if 1 > self.epsilon:    # 这个就是根据计算出的概率选择动作action
            action1 = prob_Q.max(1)[1]
            action = [item.cpu().detach().numpy() for item in action1]  # 从gpu上取下 下一个推荐项目
            action = [item.tolist() for item in action]  # 将np数据转化为list，大小为batch_size个
        else:
            action = np.random.randint(0, self.n_node, [self.batch_size]).tolist()
        return action

    def gen_reward(self, target_item, prob_Q):
        action = self.choice_action(prob_Q)
        reward = dcg_k(target_item, action, 1)  # -1 / 4 的奖励值
        return reward

    def forward(self, policy_QS, target_Qs, state, len_state, action, discount):  # 输入的就是数据，
        """
        :param policy_QS: 这是策略网络的Q值
        :param target_Qs: 这是对象网络的Q值
        :param state: 这是前一个状态，Q网络里要根据这个状态计算出output1和output2
        :param len_state: 这是前一个状态的长度（就是有效的项目数）
        :param action: 这是在当前状态转移到下一个状态时应该点击的项目，其实就是当前状态的标签
        :param reward: 这是当前状态转移到下一个状态时的奖励（在diginetica中不能作为输入，因为没有购买这个属性）
        :param discount: 这是计算奖励的折扣系数，就是公式里的gamma
        :return: 返回Double Q-network 计算出的loss值 + 推荐模型DHCN得到的loss值 的总和
        """
        # q_tm1, a_tm1, r_t, pcont_t, q_t_value, q_t_selector,
        # output1, actions, reward, discount, targetQs_, targetQs_selector
        # scores, action, reward, discount, target_Qs, policy_QS
        target_action = trans_to_cuda(torch.tensor(action))

        # 计算当前状态s_t下，policy网络选择动作的概率，即Q（s_t, a），并选出当前batch中模型得到的项目被选中最高的概率q_value。
        state = get_data(state)
        policy_value, recom_scores = self.gen_rec(state, len_state, A_hat=2)    # [B, n_node] 这个就是output2

        # reward2 = trans_to_cuda(torch.tensor(self.gen_reward(action, policy_value)))
        with torch.no_grad():
            reward = trans_to_cuda(torch.tensor(self.gen_reward(action, policy_value)))

        action = trans_to_cuda(torch.tensor(action).long().reshape(self.batch_size, 1))
        qa_tm1 = torch.gather(policy_value, dim=1, index=action).flatten()  # 当前状态下的Q值

        best_action = policy_QS.max(1)[1].reshape(self.batch_size, 1)      # 取出policy网络在s_t+1状态下，选择概率最大的动作a*。（从0开始标的号）
        q_b_value = torch.gather(target_Qs, dim=1, index=best_action).flatten()  # 当前状态下的Q值

        target = reward + discount * q_b_value
        # qa_tm1 表示的是策略网络在当前状态下采取标签动作的概率
        # target 表示的是对象网络在在下一个状态中选取当前状态下策略网络采取的最优动作的概率 + 当前状态下采取最优动作的奖励
        loss_RL = torch.mean(0.5 * torch.square(torch.sub(target.detach(), qa_tm1)))
        # loss_RL = 0.5 * nn.MSELoss()(target.detach(), qa_tm1)  # 计算 一步TD误差，即计算当前状态的Q值和上一个状态Q值的差异
        # print(loss_RL)
        loss_rec = self.loss_function(recom_scores + 1e-8, target_action)
        # print(loss_rec)
        with open("log/q_b_value.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write(str(q_b_value) + "\n")
        with open("log/reward.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write(str(reward) + "\n")
        with open("log/qa_tm1.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write(str(qa_tm1) + "\n")
        with open("log/loss_RL.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write(str(loss_RL) + "\n")
        with open("log/loss_rec.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write(str(loss_rec) + "\n")
        # with open("log/ssl_loss.txt", "a") as f:  # 记录每个epoch训练的输出，
        #     f.write(str(ssl_loss) + "\n")
        with open("log/best_action.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write(str(best_action.squeeze(-1)) + "\n")
        self.loss = loss_rec + 0.1 * loss_RL
        self.loss.backward()
        self.optimizer.step()
        return self.loss


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid, score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids   # , k_largest_scores


def forward(policy_net, target_net, data, name=None):
    """
    :param QN_1: 这个是策略网络policy_net，更新的就是这个
    :param QN_2: 这个是对象网络target_net
    :param data: 这个是当前batch的数据
    :param name: 指定当前batch是训练还是测试模块
    :return:
    """
    # 这里是第一次读取数据集中的数据，并转换为list类型的数据
    next_state = list(data['next_state'].values())
    len_next_state = list(data['len_next_states'].values())
    state = list(data['state'].values())
    len_state = list(data['len_state'].values())
    action = list(data['action'].values())  # 这个action是强化学习每次交互要预测的标签
    is_done = list(data['is_done'].values())  # 这个是强化学习中一个episode结束的标志，其实就是表示已经交互到当前会话的最后一个项目
    for i in range(len(action)):    # 数据集中的action项目标号是从1开始的，标签中的项目标号是从0开始。
        action[i] -= 1

    if name == "train":
        pointer = np.random.randint(0, 2)
        # 更新的是policy_net的参数，DDQN的结构
        if pointer == 0:
            QN_1 = policy_net
            QN_2 = target_net
        else:
            QN_1 = target_net
            QN_2 = policy_net
        # QN_1 = policy_net
        # QN_2 = target_net
        next_state = get_data(next_state)   # 对输入的数据进行处理，得到超图嵌入需要的数据形式[items, reversed_sess_item, mask]
        with torch.no_grad():
            policy_QS, _ = QN_1.gen_rec(next_state, len_next_state, A_hat=1)    # 计算在下一个状态下策略网络对应的Q值,是个在gpu上的张量
            target_Qs, _ = QN_2.gen_rec(next_state, len_next_state, A_hat=1)    # 计算在下一个状态下对象网络对应的Q值,是个在gpu上的张量
        # print(policy_QS)
        # print(target_Qs)
        # print(target_Qs.shape)  # torch.Size([BS, n_node])，表示的是当前batch中每个会话采取所有项目的概率

        # Set target_Qs to 0 for states where episode ends

        for index in range(target_Qs.shape[0]):
            if is_done[index]:
                target_Qs[index] = torch.zeros([QN_1.n_node])
                # 如果当前状态是最后一个状态，这个状态就不用计算概率了，因此全部赋值为0.
                # 就是说当前状态已经交互到这个会话的最后一个项目，那么交互完成，也就没有采样的概率
        # print(action)
        # is_buy = list(data['is_buy'].values())
        # reward = []
        # print("state = ", state)
        # print(len_state)
        # for k in range(len(is_buy)):    # len(is_buy) = 100
        #     reward.append(QN_1.reward_buy if is_buy[k] == 1 else QN_1.reward_click)
        # 这个奖励不就是直接读取数据集吗？买了就是buy的奖励值=1，没买就是click的奖励值=0.2
        # discount = [QN_1.discount] * len(action)  # 这是一个长度为100的列表，每个值都是discount
        discount = QN_1.discount

        # 求loss
        loss = QN_1(policy_QS, target_Qs, state, len_state, action, discount)

        with open("log/loss.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write(str(loss) + "\n")
    else:
        with torch.no_grad():
            action = trans_to_cuda(torch.tensor(action))
            state = get_data(state)  # 对输入的数据进行处理，得到超图嵌入需要的数据形式
            rec_scores = policy_net.gen_rec(state, len_state)  # 计算在下一个状态下策略网络对应的Q值,是个在gpu上的张量
            return action, rec_scores
    return loss


def _soft_update_q_network_parameters(q_network_1, q_network_2, alpha=1e-3):
    """In-place, soft-update of q_network_1 parameters with parameters from q_network_2."""
    for p1, p2 in zip(q_network_1.parameters(), q_network_2.parameters()):
        p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)


def train_test(QN_1, QN_2, train_data, test_data, name=None):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0

    num_rows = train_data.shape[0]  # 获取训练数集有多少条数
    # print("train num_rows = ", num_rows)    # train num_rows =  719470
    # num_rows = 2000
    num_batches = int(num_rows/QN_1.batch_size)   #+

    for j in range(num_batches):
        QN_1.optimizer.zero_grad()
        QN_2.optimizer.zero_grad()
        batch = train_data[j * QN_1.batch_size:(j+1) * QN_1.batch_size].to_dict()
        # batch1 = train_data.sample(n=QN_1.batch_size).to_dict()  # 不重复对数据集进行采样
        # print("不打乱读取：", batch)
        # print("打乱读取：", batch1)
        loss = forward(QN_1, QN_2, batch, name="train")
        # _soft_update_q_network_parameters(QN_2, QN_1)
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    QN_1.eval()
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    num_rows = test_data.shape[0]  #
    # print("test num_rows = ", num_rows)   # test num_rows =  60858
    # num_rows = 1000
    num_batches = int(num_rows/QN_1.batch_size)   # 以100为batch的大小，就有9249个batch，就有9249个loss值
    for j in range(num_batches):
        batch = test_data.sample(n=QN_1.batch_size).to_dict()
        tar, scores = forward(QN_1, QN_2, batch, name="test")
        scores = trans_to_cpu(scores).detach().numpy()
        tar = trans_to_cpu(tar).detach().numpy()
        # print(scores)
        # print(tar)
        index = []
        for idd in range(QN_1.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
        # print(metrics)
    return metrics, total_loss


def dcg_k(actual, predicted, topk):
    with open('log/r_actual.txt', 'a+') as f:
        f.write(str(actual) + '\n')
    with open('log/r_predicted.txt', 'a+') as f:
        f.write(str(predicted) + '\n')
    # (2048, 3)， (2048, 3)， 3
    k = min(topk, len(actual))
    dcgs = []
    actual = np.reshape(actual, (len(actual), 1))
    predicted = np.reshape(np.array(predicted), (len(predicted), 1))
    # print(predicted)
    # actual = actual.cpu().numpy()
    # predicted = predicted.cpu().numpy()
    # print("len(actual) = ", len(actual))
    # print("actual.shape = ", actual.shape)
    for user_id in range(len(actual)):
        value = []
        # print("predicted[user_id] =", predicted[user_id])
        for i in predicted[user_id]:
            try:
                value += [topk - int(np.argwhere(actual[user_id] == i))]
                # print("int(np.argwhere(actual[user_id] == i)) = ", np.argwhere(actual[user_id] == i))
                # print("value = ", value)
            except:
                value += [0]
        # dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(k)])
        dcg_k = sum([value[j] / math.log(j+2, 2) for j in range(k)])
        if dcg_k == 0:
            dcg_k = -1
        else:
            dcg_k += 3
        dcgs.append(dcg_k)
    return dcgs


def NDCG(rankedList, testList):
    Len_T = len(testList)
    # tar = []
    # for i in range(3):
    #     tar.append(rankedList)
    # print("testList = ", testList)
    # Len_T = 1
    with open('log/gen_target.txt') as f:
        f.write(str(rankedList) + '\n')
    with open('log/gen_predict.txt') as f:
        f.write(str(testList) + '\n')
    # print("testList = ", testList)
    # print("rankedList = ", rankedList)
    NDCG_i = 0
    for i in range(len(rankedList)):
        for j in range(len(testList[i])):
            # if rankedList[i] in testList[i]:
            if rankedList[i] or rankedList[i]+1 or rankedList[i]-1 == testList[i][j]:
                # 注意j的取值从0开始
                NDCG_i += 1 / (math.log2(1 + j + 1))
                break
    # for i in range(len(tar)):
    #     if testList[i] == tar[i]:
    #         # 注意j的取值从0开始
    #         NDCG_i += 1 / (math.log2(1 + 1))
    #         break
    NDCG_i /= Len_T
    # print(f'NDCG@5={NDCG_i}')
    return NDCG_i
