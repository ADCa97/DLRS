import pandas as pd
import numpy as np
import warnings
import random, math, os
from sklearn.utils import all_estimators
from tqdm import tqdm
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# 召回率
# 推荐系统推荐正确的物品数量占用户实际点击的物品数量的比率
def Recall(Rec_dict, Val_dict):
    """
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...} 
    Val_dict: 用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    """
    hit_items = 0
    all_items = 0
    for uid, items in Val_dict.items():
        rel_set = items
        rec_set = Rec_dict[uid]
        for item in rec_set:
            if item in rel_set:
                hit_items += 1
        all_items += len(rel_set)
    return round(hit_items / all_items * 100, 2)

# 精确率
# 推荐系统推荐正确的商品数量占给用户实际推荐的商品数量的比率
def Precision(Rec_dict, Val_dict):
    """
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...} 
    Val_dict: 用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    """
    hit_items = 0
    all_items = 0
    for uid, items in Val_dict.items():
        rel_set = items
        rec_set = Rec_dict[uid]
        for item in rec_set:
            if item in rel_set:
                hit_items += 1
        all_items += len(rec_set)
    return round(hit_items / all_items * 100, 2)

# 覆盖率
# 所有被推荐的用户中，推荐的商品数量占这些用户实际点击的商品数量的比率
def Converage(Rec_dict, Trn_dict):
    """
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...} 
    Trn_dict: 训练集用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    """
    rec_items = set()
    all_items = set()
    for uid in Rec_dict:
        for item in Trn_dict[uid]:
            all_items.add(item)
        for item in Rec_dict[uid]:
            rec_items.add(item)
    return round(len(rec_items) / len(all_items) * 100, 2)

# 使用平均流行度度量新颖度
# 如果平均流行度很高（即推荐的商品比较热门），说明推荐的新颖度比较低
def Popularity(Rec_dict, Trn_dict):
    pop_items = {}
    for uid in Trn_dict:
        for item in Trn_dict[uid]:
            if item not in pop_items:
                pop_items[item] = 0
            pop_items[item] += 1
    pop, num = 0, 0
    for uid in Rec_dict:
        for item in Rec_dict[uid]:
            pop += math.log(pop_items[item] + 1)
            num += 1
    return round(pop / num, 3)

# 评价指标
def rec_eval(val_rec_items, val_user_items, trn_user_items):
    print("Recall: ", Recall(val_rec_items, val_user_items))
    print("Precision: ", Precision(val_rec_items, val_user_items))
    print("Coverage: ", Converage(val_rec_items, trn_user_items))
    print("Popularity: ", Popularity(val_rec_items, trn_user_items))

def get_data(root_path):
    rnames = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(os.path.join(root_path, "ratings.dat"), sep="::", engine="python", names=rnames)

    # 分割训练和验证集
    trn_data, val_data, _, _ = train_test_split(ratings, ratings, test_size=0.2)

    trn_data = trn_data.groupby("user_id")["movie_id"].apply(list).reset_index()
    val_data = val_data.groupby("user_id")["movie_id"].apply(list).reset_index()

    trn_user_items = {}
    val_user_items = {}

    for user, movies in zip(*(list(trn_data["user_id"]), list(trn_data["movie_id"]))):
        trn_user_items[user] = set(movies)
    for user, movies in zip(*(list(val_data["user_id"]), list(val_data["movie_id"]))):
        val_user_items[user] = set(movies)
    
    return trn_user_items, val_user_items

def Item_CF_Rec(trn_user_items, val_user_items, K, N):
    """
    trn_user_items: 表示训练数据，格式为:{user_id1: [item_id1, item_id2,...,item_idn], user_id2...}
    val_user_items: 表示验证数据，格式为:{user_id1: [item_id1, item_id2,...,item_idn], user_id2...}
    K: K表示的是相似物品的数量，每个用户交互的每个物品都选择与其最相似的K个物品
    N: N表示的是给用户推荐的商品数量，给每个用户推荐相似度最大的N个商品
    """
    # 建立user->items倒排表，即trn_user_items

    # 计算物品协同过滤矩阵
    sim = {} # 记录物品i和j之间共同交互的用户数量
    num = {} # 记录每个物品所交互的用户数量
    for uid, items in tqdm(trn_user_items.items(), desc="构建协同过滤矩阵>>>"):
        for i in items:
            if i not in num:
                num[i] = 0
            num[i] += 1 # 物品i交互过的用户数目
            if i not in sim:
                sim[i] = {}
            for j in items:
                if i != j:
                    if j not in sim[i]:
                        sim[i][j] = 0
                    sim[i][j] += 1 # 同时交互过物品i和j的用户+1
    # 计算相似度矩阵
    for i, items in tqdm(sim.items(), desc="计算相似度>>>"):
        for j, score in items.items():
            sim[i][j] = score / math.sqrt(num[i] * num[j]) # 余弦相似度

    # TopN推荐
    items_rank = {}
    for u, _ in tqdm(val_user_items.items(), desc="TopN推荐>>>"):
        items_rank[u] = {}
        for hist_item in trn_user_items[u]: # 用户u交互过的一个历史物品
            # 计算与该物品相似的其他物品的评分
            for item, score in sorted(sim[hist_item].items(), key=lambda x: x[1], reverse=True)[:K]:
                if item not in trn_user_items[u]: # 测试用户u未交互的物品进行推荐，计算对该商品的评分
                    if item not in items_rank[u]:
                        items_rank[u][item] = 0
                    items_rank[u][item] += score # 由于是0，1的交互记录，只需要相加即可
    # 对物品进行排序
    items_rank= {k: sorted(v.items(), key = lambda x: x[1], reverse=True)[:N] for k, v in items_rank.items()}
    items_rank = {k: set([x[0] for x in v]) for k, v in items_rank.items()}
    return items_rank
if __name__ == "__main__":
    root_path = '../base_data/ml-1m/'
    trn_user_items, val_user_items = get_data(root_path)
    rec_items = Item_CF_Rec(trn_user_items, val_user_items, 80, 10)
    rec_eval(rec_items, val_user_items, trn_user_items)