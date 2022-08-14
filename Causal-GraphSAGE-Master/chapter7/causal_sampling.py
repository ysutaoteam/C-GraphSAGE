from chapter7.Data import CoraData
import numpy as np
import torch as th

DEVICE = "cuda" if th.cuda.is_available() else "cpu"
data = CoraData().data
def sampling(src_nodes, sample_num, neighbor_table, con):

    results = []
    for sid in src_nodes:
        res = np.random.choice(neighbor_table[sid], size=(sample_num,))
        neighbor_node = np.asarray(res).flatten()
        con_us = []
        for i in neighbor_node:
            con_us.append(con[i])
        label_Y = data.y[neighbor_node]
        label_yr = np.unique(label_Y)
        # ur
        con_ur = np.unique(con_us)
        hopk_result1 = cb_backdoor(neighbor_node, label_Y, con_us, label_yr, con_ur, neighbor_node.size)
        results.append(hopk_result1)
    return np.asarray(results).flatten()





def multihop_sampling(src_nodes, sample_nums, neighbor_table,con_us):

    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table, con_us)
        sampling_result.append(hopk_result)
    return sampling_result


def rand_cat_fast(p, N):
    K = len(p)
    u = np.random.rand(N, 1)
    P = np.cumsum(p, axis=0)
    U = np.tile(P.T, (N, 1))
    c = np.tile(u, (1, K)) >= U
    c = c + 0
    x = np.sum(c, 1) + 1
    return x


def histcnd(Y, U, yr, ur):
    Nyu = np.zeros((7, 2), dtype=int)
    for index, i in enumerate(Y):
        Nyu[i - 1][U[index] - 1] += 1

    return Nyu


def cb_backdoor(X, Y, U, yr, ur, M):
    #  1.Estimate f(u,y), f(y) and f(u|y)
    N = len(X)
    K = yr.shape[0]
    Nyu = histcnd(Y, U, yr, ur)
    # print(Nyu)
    pyu = Nyu / N
    pu = np.sum(pyu, axis=0, keepdims=True).T
    py = np.sum(pyu, axis=1, keepdims=True)
    py_u = pyu / pu.T
    Ns = np.sum(Nyu, axis=1)



    # 2. For each y in range of values of Y variable


    v = np.zeros((M, 1), dtype=float)

    index_i = []

    for i in yr:
        # 3. Resample indices
        for index, j in enumerate(Y):

            if j == i:
                v[index][0] = j / (py_u[i - 1][U[index] - 1]).T / N

        a = rand_cat_fast(v, Ns[i - 1])
        for i in a:
            index_i.append(i)

    x1=[]

    for i in index_i:
        if i == len(index_i) +1:
           x1.append(X[len(index_i)-1])
        else:
           x1.append(X[i - 1])
    Xw = np.array(x1)

    return Xw


