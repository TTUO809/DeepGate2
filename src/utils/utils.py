from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import math
import copy
import torch
import random
import numpy as np

from .circuit_utils import random_pattern_generator, logic

def rename_node(x_data):
    for idx in range(len(x_data)):
        x_data[idx][0] = idx
    return x_data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def zero_normalization(x):
    mean_x = torch.mean(x)
    std_x = torch.std(x)
    z_x = (x - mean_x) / std_x
    return z_x

class custom_DataParallel(nn.parallel.DataParallel):
# define a custom DataParallel class to accomodate igraph inputs
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(custom_DataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        # to overwride nn.parallel.scatter() to adapt igraph batch inputs
        G = inputs[0]
        scattered_G = []
        n = math.ceil(len(G) / len(device_ids))
        mini_batch = []
        for i, g in enumerate(G):
            mini_batch.append(g)
            if len(mini_batch) == n or i == len(G)-1:
                scattered_G.append((mini_batch, ))
                mini_batch = []
        return tuple(scattered_G), tuple([{}]*len(scattered_G))

def collate_fn(G):
    return [copy.deepcopy(g) for g in G]

def pyg_simulation(g, pattern=[]):
    # PI, Level list
    max_level = 0
    PI_indexes = []
    fanin_list = []
    for idx, ele in enumerate(g.forward_level):
        level = int(ele)
        fanin_list.append([])
        if level > max_level:
            max_level = level
        if level == 0:
            PI_indexes.append(idx)
    level_list = []
    for level in range(max_level + 1):
        level_list.append([])
    for idx, ele in enumerate(g.forward_level):
        level_list[int(ele)].append(idx)
    # Fanin list 
    for k in range(len(g.edge_index[0])):
        src = g.edge_index[0][k]
        dst = g.edge_index[1][k]
        fanin_list[dst].append(src)
    
    ######################
    # Simulation
    ######################
    y = [0] * len(g.x)
    if len(pattern) == 0:
        pattern = random_pattern_generator(len(PI_indexes))
    j = 0
    for i in PI_indexes:
        y[i] = pattern[j]
        j = j + 1
    for level in range(1, len(level_list), 1):
        for node_idx in level_list[level]:
            source_signals = []
            for pre_idx in fanin_list[node_idx]:
                source_signals.append(y[pre_idx])
            if len(source_signals) > 0:
                if int(g.x[node_idx][1]) == 1:
                    gate_type = 1
                elif int(g.x[node_idx][2]) == 1:
                    gate_type = 5
                else:
                    raise("This is PI")
                y[node_idx] = logic(gate_type, source_signals)

    # Output
    if len(level_list[-1]) > 1:
        raise('Too many POs')
    return y[level_list[-1][0]], pattern

def get_function_acc(g, node_emb):
    MIN_GAP = 0.05
    # Sample
    retry = 10000
    tri_sample_idx = 0
    correct = 0
    total = 0
    while tri_sample_idx < 100 and retry > 0:
        retry -= 1
        sample_pair_idx = torch.LongTensor(random.sample(range(len(g.tt_pair_index[0])), 2))
        pair_0 = sample_pair_idx[0]
        pair_1 = sample_pair_idx[1]
        pair_0_gt = g.tt_dis[pair_0]
        pair_1_gt = g.tt_dis[pair_1]
        if pair_0_gt == pair_1_gt:
            continue
        if abs(pair_0_gt - pair_1_gt) < MIN_GAP:
            continue

        total += 1
        tri_sample_idx += 1
        pair_0_sim = torch.cosine_similarity(node_emb[g.tt_pair_index[0][pair_0]].unsqueeze(0), node_emb[g.tt_pair_index[1][pair_0]].unsqueeze(0), eps=1e-8)
        pair_1_sim = torch.cosine_similarity(node_emb[g.tt_pair_index[0][pair_1]].unsqueeze(0), node_emb[g.tt_pair_index[1][pair_1]].unsqueeze(0), eps=1e-8)
        pair_0_predDis = 1 - pair_0_sim
        pair_1_predDis = 1 - pair_1_sim
        succ = False
        if pair_0_gt > pair_1_gt and pair_0_predDis > pair_1_predDis:
            succ = True
        elif pair_0_gt < pair_1_gt and pair_0_predDis < pair_1_predDis:
            succ = True
        if succ:
            correct += 1

    if total > 0:
        acc = correct * 1.0 / total
        return acc
    return -1
            
def generate_orthogonal_vectors(n, dim):
    # Generate an initial random vector
    v0 = np.random.randn(dim)
    v0 /= np.linalg.norm(v0)

    # Generate n-1 additional vectors
    vectors = [v0]
    for i in range(n-1):
        # Generate a random vector
        v = np.random.randn(dim)

        # Project the vector onto the subspace spanned by the previous vectors
        for j in range(i+1):
            v -= np.dot(v, vectors[j]) * vectors[j]

        # Normalize the vector
        v /= np.linalg.norm(v)

        # Append the vector to the list
        vectors.append(v)

    # calculate the max cosine similarity between any two vectors
    max_cos_sim = 0
    for i in range(n):
        for j in range(i+1, n):
            vi = vectors[i]
            vj = vectors[j]
            cos_sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
            if cos_sim > max_cos_sim:
                max_cos_sim = cos_sim

    return vectors, max_cos_sim

# 定義一個函數來生成初始的 pi 向量，並計算它們之間的最大相似度
def generate_hs_init(G, hs, no_dim, homo_pi_init=False):
    # G: 圖結構資料（含節點資訊）；hs: 節點隱藏狀態張量；
    # no_dim: 向量維度；homo_pi_init: 是否使用同質初始化
    max_sim = 0  # 初始化整體最大相似度為 0
    if G.batch == None:
        # 若沒有 batch 資訊，表示只有一張圖
        batch_size = 1
    else:
        # 取 batch 中最大的索引值加 1，得到批次內的圖數量
        batch_size = G.batch.max().item() + 1
    for batch_idx in range(batch_size):  # 逐一遍歷每張圖
        if G.batch == None:
            # 單圖情況：找出所有 forward_level 為 0 的節點（即主要輸入節點 PI）
            pi_mask = (G.forward_level == 0)
        else:
            # 多圖情況：同時篩選屬於當前批次且 forward_level 為 0 的節點
            pi_mask = (G.batch == batch_idx) & (G.forward_level == 0)
        pi_node = G.forward_index[pi_mask]  # 取出 PI 節點的實際索引
        if homo_pi_init:
            # 同質初始化：所有 PI 節點共用同一個單位向量（各維度均等分配）
            shared = np.ones(no_dim, dtype=np.float32) / np.sqrt(no_dim)
            # 將相同的向量複製給每個 PI 節點
            pi_vec = [shared for _ in range(len(pi_node))]
            # 所有向量完全相同，相似度為 1.0
            batch_max_sim = 1.0
        else:
            # 非同質初始化：呼叫函數生成盡量互相正交的向量，並回傳最大相似度
            pi_vec, batch_max_sim = generate_orthogonal_vectors(len(pi_node), no_dim)
        if batch_max_sim > max_sim:
            # 更新全域最大相似度（取所有批次中的最大值）
            max_sim = batch_max_sim
        # 將生成的向量指派給對應 PI 節點的隱藏狀態
        hs[pi_node] = torch.tensor(pi_vec, dtype=torch.float)

    return hs, max_sim  # 回傳更新後的隱藏狀態與最大相似度
