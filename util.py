import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter
import operator
import copy
import itertools


def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)



def get_sub_graph(max_n_node, sess, A, alias_inputs):
    # print(np.nonzero(sess))
    # node = np.unique(np.array(sess)[np.nonzero(sess)])  # 找出每个会话中不同的项目id
    # print(sess)
    node = np.unique(sess)  # 找出每个会话中不同的项目id
    # print(node)
    # print("node : ", node)
    # print(len(node))
    # items.append(node.tolist() + (max_n_node - len(node)) * [0])  # 填充数据，不足最大长度的填0
    u_A = np.zeros((max_n_node, max_n_node))  # 创建邻接矩阵，大小为【seq_len+1,seq_len+1】,因为包含了一个0
    # print("sess: ", sess)
    for i in np.arange(len(sess) - 1):  # 遍历每条数据sess中的项目
        if sess[i + 1] == 0:
            break
        u = np.where(node == sess[i])[0][0]
        # print("u:", u)
        v = np.where(node == sess[i + 1])[0][0]
        # print("v:", v)
        u_A[u][v] = 1
    # print("u_A : ",u_A)
    u_sum_in = np.sum(u_A, 0)
    u_sum_in[np.where(u_sum_in == 0)] = 1
    u_A_in = np.divide(u_A, u_sum_in)
    u_sum_out = np.sum(u_A, 1)
    u_sum_out[np.where(u_sum_out == 0)] = 1
    u_A_out = np.divide(u_A.transpose(), u_sum_out)
    u_A = np.concatenate([u_A_in, u_A_out]).transpose()
    A.append(u_A)
    # print("A:", A)
    session_pos = [np.where(node == i)[0][0] for i in sess]  # 记录这个会话中每个项目的在node项目列表中的位置
    # session_pos = session_pos + (max_n_node - len(session_pos)) * [-1]
    alias_inputs.append(session_pos)
    # print("pos a ", session_pos)
    return A, alias_inputs

def get_data(data):
    items, num_node = [], []
    for session in data:
        num_node.append(len(np.nonzero(session)[0]))
    max_n_node = np.max(num_node)
    # print(max_n_node)
    A, alias_inputs, state, mask, reversed_sess_item = [], [], [], [], []
    for session in data:
        # print(session)
        if len(np.nonzero(session)[0]) == 0:
            print("!!!!!!!!!!")
            session[0] = np.random.randint(1, 43098)
        # session_len.append([len(nonzero_elems)])
        nonzero_elems = np.nonzero(session)[0]
        # items.append(session)
        mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
        # new_arr_0 = session[np.where(arr == 0)]
        session1 = np.array(session)
        new_arr_no_0 = session1[np.where(session1 != 0)]
        reversed_sess_item.append(list(reversed(new_arr_no_0)) + (max_n_node - len(nonzero_elems)) * [0])
        sess_data = list(new_arr_no_0) + (max_n_node - len(nonzero_elems)) * [0]
        items.append(sess_data)
        A, alias_inputs = get_sub_graph(max_n_node, sess_data, A, alias_inputs)
        # A 是【B，2*max_node，max_node】，alias_inputs是【B，每个会话的长度（没有0）】
    # print(items)
    # print(reversed_sess_item)
    # print(len(reversed_sess_item))
    state.extend([items, reversed_sess_item, mask, A, alias_inputs])
    return state


def get_overlap(sessions):
    matrix = np.zeros((len(sessions), len(sessions)))
    for i in range(len(sessions)):
        seq_a = set(sessions[i])
        seq_a.discard(0)
        for j in range(i+1, len(sessions)):
            seq_b = set(sessions[j])
            seq_b.discard(0)
            overlap = seq_a.intersection(seq_b)
            ab_set = seq_a | seq_b
            matrix[i][j] = float(len(overlap))/float(len(ab_set))
            matrix[j][i] = matrix[i][j]
    matrix = matrix + np.diag([1.0]*len(sessions))
    degree = np.sum(np.array(matrix), 1)
    degree = np.diag(1.0/degree)
    return matrix, degree


class Data():
    def __init__(self, data, shuffle=False, n_node=None, name=None):
        self.raw = np.asarray(data)
        # print("raw data", self.raw[:70])
        # if operator.eq(name, 'train'):
        #     for i in self.raw:
        #         if 26702 in i:    # 项目编号是从1-43097的
        #             print(i)
        #             print("项目中存在0")
        # if operator.eq(name, 'train'):
        #     print("train data:1w 条")
        #     self.raw = self.raw[:10000]
        # elif operator.eq(name, 'test'):
        #     print("test data: 1000 条")
        #     self.raw = self.raw[:1000]
        H_T = data_masks(self.raw, n_node)
        BH_T = H_T.T.multiply(1.0/H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0/H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        self.adjacency = DHBH_T.tocoo()
        # print(self.adjacency)
        self.n_node = n_node
        self.length = len(self.raw)
        self.shuffle = shuffle

    # 这个函数是生成线图用的一些信息
    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index, name="train", flag=False, sess=None):
        items, num_node = [], []
        items_nm = []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            items_nm.append(session)
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        return self.targets[index]-1, session_len, items, reversed_sess_item, mask, items_nm
