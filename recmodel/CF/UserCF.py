from tqdm import tqdm
import math

class UserCF:

    def __init__(self, train_data: dict, test_data: dict):
        """
        UserCF模型
        Args:
            train_data: dict, key = userID, value = set(movieID), user->items的倒排索引表
            test_data: dict, key = userID, value = set(movieID), user->items的倒排索引表
        """
        # 建立item->users倒排表
        itemUsers = {}
        for userid, items in tqdm(train_data.items(), desc="建立item->users倒排表>>>"):
            for itemid in items:
                if itemid not in itemUsers:
                    itemUsers[itemid] = set()
                itemUsers[itemid].add(userid)
        self.train_userItems = train_data
        self.train_itemUsers = itemUsers
        self.test_userItems = test_data
        # 计算相似度矩阵
        self.sim = self.calsimilarity()


    def calsimilarity(self) -> dict:
        """
        Returns:
            similarity: dict, 相似度矩阵, sim[i][j]代表user_i和user_j之间的相似度
        """
        sim = {} # 记录user_i和user_j之间共同交互过的项目数量, 对应user-item共现矩阵行向量的内积
        num = {} # 记录user_i所交互过的项目数量, 对应user-item共现矩阵行向量的大小
        for itemid, users in tqdm(self.train_itemUsers.items(), desc="构建协同过滤矩阵>>>"):
            for user_i in users:
                # user_i交互过的项目数量+1
                if user_i not in num:
                    num[user_i] = 0
                num[user_i] += 1
                # 两两user之间点击过同一个项目
                if user_i not in sim:
                    sim[user_i] = {}
                for user_j in users:
                    if user_i != user_j:
                        if user_j not in sim[user_i]:
                            sim[user_i][user_j] = 0
                        sim[user_i][user_j] += 1
        # 计算相似度矩阵
        for user_i, users in tqdm(sim.items(), desc="计算相似度>>>"):
            for user_j, score in users.items():
                sim[user_i][user_j] = score / math.sqrt(num[user_i] * num[user_j]) # 余弦相似度
        return sim
    def rec(self, K: int, N: int) -> dict:
        """
        Args:
            K: int, 为测试用户u选择与其最相似的K个用户
            N: int, 给用户推荐的项目数量N
        Returns:
            items_rank: dict, items_rank[u]表示为用户u推荐的N个item集合
        """
        items_rank = {}
        for u, _ in tqdm(self.test_userItems.items(), desc="TopN推荐>>>"):
            items_rank[u] = {}
            for v, score in sorted(self.sim[u].items(), key=lambda x: x[1], reverse=True)[:K]: # 选择最相似的K个用户
                for item in self.train_userItems[v]: # 遍历用户v交互过的项目
                    if item not in self.train_userItems[u]: # 其中测试用户u未交互过的项目
                        if item not in items_rank[u]:
                            items_rank[u][item] = 0
                        items_rank[u][item] += score

        # 对item排序
        items_rank = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:N] for k, v in items_rank.items()}
        items_rank = {k: set(x[0] for x in v) for k, v in items_rank.items()}
        return items_rank





        



    