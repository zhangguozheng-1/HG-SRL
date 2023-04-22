import argparse
import pickle
import time
from util import Data, split_validation
from RL import *
import os
import numpy as np
import pandas as pd
# from torch.nn import DataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/Nowplaying/sample/RC15')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=200, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

# parser = add_argument(description="Run nive double q learning.")
parser.add_argument('--hidden_factor', type=int, default=64, help='Number of hidden factors, i.e., embedding size.')
parser.add_argument('--r_click', type=float, default=0.2, help='reward for the click behavior.')
parser.add_argument('--r_buy', type=float, default=1.0, help='reward for the purchase behavior.')
parser.add_argument('--discount', type=float, default=0.5, help='Discount factor for RL.')
parser.add_argument('--state_size', type=int, default=10, help='each session length.')

opt = parser.parse_args()
print(opt)
gpus = [0, 1, 2]

def main():
    # session_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    # session_data = pickle.load(open('datasets/' + opt.dataset + '/train_session.txt', 'rb'))
    # session_datas 包含两行，第一行是session，第二行是target item
    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    else:
        n_node = 309

    train_data = pd.read_pickle(os.path.join('datasets/' + opt.dataset + '/diginetica_train.df'))
    test_data = pd.read_pickle(os.path.join('datasets/' + opt.dataset + '/diginetica_test.df'))
    # train_data_num = train_data.shape[0]   #  719470
    # test_data_num = test_data.shape[0]   # 60858
    """"""
    # train_session = list(train_data.to_dict()['next_state'].values())
    train_session = list(train_data.to_dict()['state'].values())
    # print(train_data.to_dict().values())
    train_session = np.array(train_session)
    print(train_session.shape)  # (719470, 10)  # (719470, 70)
    sess_data = []
    for session in train_session:   # 读取的数据中包含填充的0，这里需要去掉，方便计算超图的邻接矩阵
        session = list(session[np.nonzero(session)])
        sess_data.append(session)

    session_data = Data(sess_data, shuffle=True, n_node=n_node, name="train")
    # sess_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    # session_data = Data1(sess_data, shuffle=True, n_node=n_node, name="train")
    # with open('rx.txt', 'a') as f:
    #     f.write(str(session_data1.adjacency))
    # with open('rx1.txt', 'a') as f:
    #     f.write(str(session_data.adjacency))
    # 强化学习Double deep Q-learning
    # QN_1 = trans_to_cuda(Qnetwork(adjacency=session_data.adjacency, n_node=n_node, lr=opt.lr, l2=opt.l2,
    #                               beta=opt.beta, layers=opt.layer, emb_size=opt.embSize, state_size=opt.state_size,
    #                               batch_size=opt.batchSize, dataset=opt.dataset,
    #                               reward_click=opt.r_click, reward_buy=opt.r_buy, discount=opt.discount))
    # QN_2 = trans_to_cuda(Qnetwork(adjacency=session_data.adjacency, n_node=n_node, lr=opt.lr, l2=opt.l2,
    #                                beta=opt.beta, layers=opt.layer, emb_size=opt.embSize, state_size=opt.state_size,
    #                                batch_size=opt.batchSize, dataset=opt.dataset,
    #                                reward_click=opt.r_click, reward_buy=opt.r_buy, discount=opt.discount))
    QN_1 = trans_to_cuda(Qnetwork(adjacency=session_data.adjacency, n_node=n_node, lr=opt.lr, l2=opt.l2,
                                  beta=opt.beta, layers=opt.layer, emb_size=opt.embSize, state_size=opt.state_size,
                                  batch_size=opt.batchSize, dataset=opt.dataset,
                                  reward_click=opt.r_click, reward_buy=opt.r_buy, discount=opt.discount))
    QN_2 = trans_to_cuda(Qnetwork(adjacency=session_data.adjacency, n_node=n_node, lr=opt.lr, l2=opt.l2,
                                   beta=opt.beta, layers=opt.layer, emb_size=opt.embSize, state_size=opt.state_size,
                                   batch_size=opt.batchSize, dataset=opt.dataset,
                                   reward_click=opt.r_click, reward_buy=opt.r_buy, discount=opt.discount))
    # QN_1 = DataParallel(QN_1, device_ids=gpus, output_device=gpus[0])
    # QN_2 = DataParallel(QN_2, device_ids=gpus, output_device=gpus[0])
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        with open("log/loss.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write("--------" + str(cur_time) + "--------" + "\n")
        metrics, total_loss = train_test(QN_1, QN_2, train_data, test_data)

        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        with open("metrics.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write(str(metrics) + "\n")
        with open("best_metric.txt", "a") as f:  # 记录每个epoch训练的输出，
            f.write(str(best_results) + "\n")
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == '__main__':
    main()
