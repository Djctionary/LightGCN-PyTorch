#!/usr/bin/env python3
import os
import pandas as pd
import random
from collections import defaultdict

def main():
    # 数据集根路径（请根据实际情况修改）
    data_path = '/home/jiacdong/Projects/LightGCN-PyTorch/data/Book'

    # 读取并保留正反馈评分
    ratings_file = os.path.join(data_path, 'Ratings.csv')
    ratings = pd.read_csv(ratings_file)
    ratings = ratings[ratings['Book-Rating'] > 0]

    # 映射原始 ID 到连续 0-base 索引
    unique_user_ids = ratings['User-ID'].unique()
    unique_item_ids = ratings['ISBN'].unique()
    user2idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    item2idx = {iid: idx for idx, iid in enumerate(unique_item_ids)}

    # 确保输出目录存在
    os.makedirs(data_path, exist_ok=True)

    # 保存映射表为 CSV 文件
    user_list_file = os.path.join(data_path, 'user_list.csv')
    pd.DataFrame(list(user2idx.items()), columns=['org_id', 'remap_id']) \
        .to_csv(user_list_file, index=False)
    item_list_file = os.path.join(data_path, 'item_list.csv')
    pd.DataFrame(list(item2idx.items()), columns=['org_id', 'remap_id']) \
        .to_csv(item_list_file, index=False)

    # 应用映射到 Ratings DataFrame
    ratings['user_idx'] = ratings['User-ID'].map(user2idx)
    ratings['item_idx'] = ratings['ISBN'].map(item2idx)

    # 构建每个用户的物品交互列表
    user_hist = defaultdict(list)
    for row in ratings.itertuples(index=False):
        user_hist[row.user_idx].append(row.item_idx)

    # 固定随机种子以便复现
    random.seed(2020)

    train_data = []
    test_data = []
    # 划分训练/测试集
    for u, items in user_hist.items():
        n = len(items)
        if n == 0:
            continue
        if n == 1:
            # 只有一次交互，仅保留到测试集，不参与训练
            test_data.append((u, items))
            continue
        # 多次交互用户：按 80/20 划分
        test_size = max(1, int(n * 0.2))
        if test_size >= n:
            test_size = n - 1
        test_items = random.sample(items, test_size)
        train_items = [i for i in items if i not in test_items]
        train_data.append((u, train_items))
        test_data.append((u, test_items))

    # 写入 train.txt 与 test.txt，空格分隔
    train_file = os.path.join(data_path, 'train.txt')
    test_file  = os.path.join(data_path, 'test.txt')
    with open(train_file, 'w') as f:
        for u, items in train_data:
            f.write(f"{u} {' '.join(map(str, items))}\n")
    with open(test_file, 'w') as f:
        for u, items in test_data:
            if items:
                f.write(f"{u} {' '.join(map(str, items))}\n")
            else:
                f.write(f"{u}\n")

if __name__ == '__main__':
    main()
