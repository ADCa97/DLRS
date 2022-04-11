from ast import arg
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse

def preProcessData(dataPath: str = "../data/ml-1m/") -> tuple:
    """
    预处理数据集, 根据用户ID按比例划分训练集与测试集.

    Args:
        dataPath: string, the directory of data
    Returns:
        train_data: dict, key = userID, item = set(movieID)
        test_data: dict, key = userID, item = set(movieID)
    """
    head = ["userID", "movieID", "rating", "timeStamp"]
    ratings = pd.read_csv(os.path.join(dataPath, "ratings.dat"), sep="::", engine="python", names=head , usecols=[0, 1, 2])

    # 分割训练集和测试集,stratify=ratings["userID"]确保训练集和测试集中用户比例一致
    train_data, test_data, _, _ = train_test_split(ratings, ratings, test_size=0.2, random_state=97, stratify=ratings["userID"])

    # 按userID分组
    train_data = train_data.groupby("userID")["movieID"].apply(list).reset_index()
    test_data = test_data.groupby("userID")["movieID"].apply(list).reset_index()

    # user->items倒排索引表
    train_userItems = {}
    test_userItems = {}

    for user, movies in zip(*(list(train_data["userID"]), list(train_data["movieID"]))):
        train_userItems[user] = set(movies)
    for user, movies in zip(*(list(test_data["userID"]), list(test_data["movieID"]))):
        test_userItems[user] = set(movies)
    return train_userItems, test_userItems




def parse_args():
    parser = argparse.ArgumentParser(description='Collaborative Filter')
    parser.add_argument("--model", 
                        type=str, 
                        default="itemcf", 
                        help="use which algorithm to recommend")
    parser.add_argument("--K",
                        type=int,
                        default=1,
                        help="choose K neighbors to calculate")
    parser.add_argument("--N",
                        type=int,
                        default=10,
                        help="choos N items to recommend")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 读取数据集
    train_data, test_data = preProcessData("../data/ml-1m/")



  