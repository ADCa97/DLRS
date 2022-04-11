from tqdm import tqdm
import math

class ItemCF:

    def __init__(self, train_data: dict, test_data: dict):
        """
        ItemCF模型
        Args:
            train_data: dict, key = userID, value = set(movieID), user->items的倒排索引表
            test_data: dict, key = userID, value = set(movieID), user->items的倒排索引表
        """
        self.train_userItems = train_data
        self.test_userItems = test_data
        # 计算相似度矩阵
        self.sim = self.calsimilarity()


    def calsimilarity(self) -> dict:
        """
        Returns:
            similarity: dict, 相似度矩阵, sim[i][j]代表item_i和item_j之间的相似度
        """
        sim = {} # 记录item_i和item_j之间共同交互过的用户数量, 对应user-item共现矩阵列向量的内积
        num = {} # 记录item_i所交互过的用户数量, 对应user-item共现矩阵列向量的大小
        for userid, items in tqdm(self.train_userItems.items(), desc="构建协同过滤矩阵>>>"):
            for item_i in items:
                # item_i交互过的用户数量+1
                if item_i not in num:
                    num[item_i] = 0
                num[item_i] += 1
                # 两两item之间被同一个用户点击过
                if item_i not in sim:
                    sim[item_i] = {}
                for item_j in items:
                    if item_i != item_j:
                        if item_j not in sim[item_i]:
                            sim[item_i][item_j] = 0
                        sim[item_i][item_j] += 1
        # 计算相似度矩阵
        for item_i, items in tqdm(sim.items(), desc="计算相似度>>>"):
            for item_j, score in items.items():
                sim[item_i][item_j] = score / math.sqrt(num[item_i] * num[item_j]) # 余弦相似度
        return sim
    def rec(self, K: int, N: int) -> dict:
        """
        Args:
            K: int, 为用户交互过的每个item, 选择与其最相似的K个项目
            N: int, 给用户推荐的项目数量N
        Returns:
            items_rank: dict, items_rank[u]表示为用户u推荐的N个item集合
        """
        items_rank = {}
        for u, _ in tqdm(self.test_userItems.items(), desc="TopN推荐>>>"):
            items_rank[u] = {}
            for hist_item in self.train_userItems[u]: # 用户u历史上交互过的一个hist_item
                # 计算与hist_item相似的其他物品
                for item, score in sorted(self.sim[hist_item].items(), key=lambda x: x[1], reverse=True)[:K]: # 选择最相似的K个物品
                    if item not in self.train_userItems[u]:
                        if item not in items_rank[u]:
                            items_rank[u][item] = 0
                        items_rank[u][item] += score # 用户u对item的可能评分
        # 对item排序
        items_rank = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:N] for k, v in items_rank.items()}
        items_rank = {k: set(x[0] for x in v) for k, v in items_rank.items()}
        return items_rank





        



    