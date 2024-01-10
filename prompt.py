import jsonlines
import json 
from typing import Dict, List
import random 
import numpy as np
from tqdm import tqdm

np.random.seed(42)
random.seed(42)
def load_json(file_path:str)->Dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def generate_sequential_recommend_prompt(data):

    prompt = f"""The user has interacted with the following {data["user_history"]} and his preferences are encoded in <unk>.
    Based on this information, I want you to select the user's favorite item in the list of candidate items following: {data["candidate_list"]}. Your response should be a item's title in the list of candidate items.
"""
    return {"input":prompt, 'output': data['target_item'], "user_id":data['user_id'], "candidate_item":0, "task_type":data["task_type"]}

def generate_rating_prompt(data):

    prompt = f"""The user's preferences are encoded in the <unk> , additionally, he has recently purchased the following items he likes {data["user_likes"]} 
    Based on the user's preferences and items he liked, I want you to determine if the user will like the item whose title is  {data["candidate_title"]} with the feature <unk>. Your response should be a simple "Yes" or "No" without any explanations.
"""
    return {"input":prompt, 'output': data['target_item'], "user_id":data['user_id'], "candidate_item":data["candidate_item"], "task_type":data["task_type"]}

def data_process(user_sequeneces_title:str, user_sequences_id:str, user_rating:str):

    user_title_sequences = load_json(user_sequeneces_title)
    user_id_sequences = load_json(user_sequences_id)
    user_rating = load_json(user_rating)
    id_2_title = {}
    user_prefences = {}
    dislike_count = 0
    like_count = 0
    all_item_id = set()
    for user in user_title_sequences:
        item_ids = user_id_sequences[user][1:]
        item_titles = user_title_sequences[user]
        try:
            assert len(item_ids) == len(item_titles) == len(user_rating[user])
        except:
            continue
        for id in item_ids:
            id_2_title[id] = item_titles[item_ids.index(id)]
            all_item_id.add(id)
        user_prefences[user] = user_prefences.get(user, {})
        user_prefences[user]["user_likes"] = user_prefences[user].get("user_likes", [])
        user_prefences[user]["user_dislikes"] = user_prefences[user].get("user_dislikes", [])
        for id , rating in zip(item_ids, user_rating[user]):
            if rating < 4:
                user_prefences[user]["user_dislikes"].append(id)
                dislike_count +=1
            else:
                user_prefences[user]["user_likes"].append(id)
                like_count += 1

    return id_2_title, user_prefences, all_item_id, user_title_sequences, user_id_sequences

def generate_data(id_2_title, user_prefences, all_item_id, user_title_sequences,user_id_sequences):

    rating_train_data = []
    rating_test_data = []
    sequential_train_data = []
    sequential_test_data = []
    
    for user in tqdm(user_prefences):
        assert len(user_title_sequences[user]) == len(user_id_sequences[user])-1
        user_like_length = len(user_prefences[user]['user_likes'])
        user_dislike_length = len(user_prefences[user]['user_dislikes'])
        if user_dislike_length <= 5 or user_like_length <= 5:
            continue
    
        #### 评分预测任务数据集

        for i in range(4,0,-1):
            if i > 1:
                
                user_like_item = random.sample(user_prefences[user]['user_likes'][:-1],4)
                user_dislike_item = random.sample(user_prefences[user]['user_dislikes'][:-1],4)
                tmp = {}
                if random.random() < 0.5:
                    user_like_history = user_like_item[:-1]  
                    candidate_item = user_like_item[-1]
                    tmp["candidate_item"] = candidate_item
                    tmp["candidate_title"] = id_2_title[candidate_item]
                    user_dislike_history = user_dislike_item
                    tmp["target_item"] = "Yes."
                else:
                    user_like_history = user_like_item
                    candidate_item = random.sample(all_item_id,1)[0]
                    tmp["candidate_item"] = candidate_item
                    tmp["candidate_title"] = id_2_title[candidate_item]
                    user_dislike_history = user_dislike_item
                    tmp["target_item"] = "No."
                tmp["user_likes"] = [id_2_title[id] for id in user_like_history]
                tmp["user_dislikes"] = [id_2_title[id] for id in user_dislike_history]
                tmp["user_id"] = eval(user)
                tmp["task_type"] = "rating"
                rating_train_data.append(tmp)
            else:
                user_like_item = user_prefences[user]['user_likes'][:-1][-5:]   ### 确保喜欢和不喜欢的商品都不超过5个
                user_dislike_item = user_prefences[user]['user_dislikes'][:-1][-5:]
                if random.random() < 0.5:
                    user_like_history = user_like_item
                    candidate_item = user_prefences[user]["user_likes"][-1]
                    tmp["candidate_item"] = candidate_item
                    tmp["candidate_title"] = id_2_title[candidate_item]
                    user_dislike_history = user_dislike_item
                    tmp["target_item"] = "Yes."
                else:
                    user_like_history = user_like_item
                    candidate_item = user_prefences[user]["user_dislikes"][-1]
                    tmp["candidate_item"] = candidate_item
                    tmp["candidate_title"] = id_2_title[candidate_item]
                    user_dislike_history = user_dislike_item
                    tmp["target_item"] = "No."
                tmp["user_likes"] = [id_2_title[id] for id in user_like_history]
                tmp["user_dislikes"] = [id_2_title[id] for id in user_dislike_history]
                tmp["user_id"] = eval(user)
                tmp["task_type"] = "rating"
                rating_test_data.append(tmp)
    
    ### 序列推荐
    
        
        tmp = {}
        start_index = random.sample([-1 * i for i in range(7,15)], 1)[0]
        purchase_history = user_title_sequences[user][start_index:-3]
        candidate_item = user_id_sequences[user][-2]
        candidate_list = random.sample(list(all_item_id), 9)
       # candidate_list = [id_2_title[id] for id in candidate_list]
        candidate_list.append(candidate_item)
        random.shuffle(candidate_list)
        
        tmp["task_type"] = "sequential"
        tmp["user_history"] = purchase_history
        tmp["candidate_list"] = [id_2_title[item] for item in candidate_list]
        tmp["target_item"] = id_2_title[candidate_item]

        tmp["user_id"] = eval(user)
        sequential_train_data.append(tmp)

        purchase_history_test = user_title_sequences[user][start_index:-2]
        candidate_item_test = user_id_sequences[user][-1]
        candidate_list_test = random.sample(list(all_item_id), 9)
        #candidate_list_test = [id_2_title[id] for id in candidate_list_test]
        candidate_list_test.append(candidate_item_test)
        random.shuffle(candidate_list_test)
        
        tmp["task_type"] = "sequential"
        tmp["user_history"] = purchase_history_test
        tmp["candidate_list"] = [id_2_title[item] for item in candidate_list]
        tmp["target_item"] = id_2_title[candidate_item]
        tmp["user_id"] = eval(user)
        sequential_test_data.append(tmp)

    return rating_train_data, rating_test_data, sequential_train_data, sequential_test_data
                        

def main():   
    id_2_title, user_preference, all_item_id, user_title_sequences,user_id_sequences = data_process('../data/books/user_sequences_title.json', "../data/books/user_sequences_id.json",
             '../data/books/user_rating.json')
    rating_train_data, rating_test_data ,sequential_train_data, sequential_test_data = generate_data(id_2_title, user_preference, all_item_id,user_title_sequences,user_id_sequences)
    rating_train = []
    rating_test = []
    sequential_res = []
    sequential_test = []
    for data in rating_train_data:
        rating_train.append(generate_rating_prompt(data))
    random.shuffle(rating_train)
    print("rating train data length: ", len(rating_train))

    for data in rating_test_data:
        rating_test.append(generate_rating_prompt(data))
    random.shuffle(rating_test)
    print("rating test data length: ", len(rating_test))

    for data in sequential_train_data:
        sequential_res.append(generate_sequential_recommend_prompt(data))
    for data in sequential_test_data:
        sequential_test.append(generate_sequential_recommend_prompt(data))

    print("sequential data length: ", len(sequential_res))
    train_data = random.sample(rating_train, 8000) + random.sample(sequential_res, 2000)
    random.shuffle(train_data)
    print("train data length: ", len(train_data))
    test_data = random.sample(rating_test, 2000) + random.sample(sequential_test, 500)
    random.shuffle(test_data)
    print("test data length: ", len(test_data))
    with jsonlines.open('../data/processed_data/books_prompt_train_v3.jsonl', 'w') as writer:
        writer.write_all(train_data)
    with jsonlines.open('../data/processed_data/books_prompt_vaild_v3.jsonl', 'w') as writer:
        writer.write_all(test_data)



if __name__ == "__main__":
    main()