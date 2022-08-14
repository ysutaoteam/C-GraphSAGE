from chapter7.Data import CoraData
import numpy as np
import torch as th
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

data = CoraData().data
x = data.x / data.x.sum(1, keepdims=True)
train_index = np.where(data.train_mask)[0]
train_label = data.y[train_index]

def sampling(src_nodes, sample_num, neighbor_table):

    results = []
    for sid in src_nodes:

        res = np.random.choice(neighbor_table[sid], size=(sample_num, ))
        results.append(res)
    return np.asarray(results).flatten()



def multihop_sampling1(src_nodes, sample_nums, neighbor_table):

    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result


