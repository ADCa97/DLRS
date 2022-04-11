from utils import preProcessData, parse_args
from ItemCF import ItemCF
from UserCF import UserCF
from sklearn.metrics import accuracy_score



def Recall(Rec_dict, Val_dict):
    """
    召回率: 推荐系统推荐正确的物品数量占用户实际点击的物品数量的比率
    Args:
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

def Precision(Rec_dict, Val_dict):
    """
    精确率: 推荐系统推荐正确的商品数量占给用户实际推荐的商品数量的比率
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

def rec_eval(val_rec_items, val_user_items):
    print("Recall: ", Recall(val_rec_items, val_user_items))
    print("Precision: ", Precision(val_rec_items, val_user_items))

if __name__ == "__main__":
    args = parse_args()
    # 读取训练集和测试集
    train_data, test_data = preProcessData("../data/ml-1m/")
    if (args.model == "itemcf"):
        model = ItemCF(train_data, test_data)
    else:
        model = UserCF(train_data, test_data)
    items_rank = model.rec(args.K,args.N)
    rec_eval(items_rank, test_data)
    

