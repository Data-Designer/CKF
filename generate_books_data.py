import jsonlines
import json 
from typing import Dict, List
import random 
import numpy as np
from torch import exp_, rand
from tqdm import tqdm
from copy import deepcopy

np.random.seed(42)
random.seed(42)

def load_json(file_path:str)->Dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def generate_sequential_recommend_prompt(data):

    
    prompt = f"""The user's preferences are encoded in <unk>, he has interacted with the following items:{data["history_list"]} recently.
    Based on this information, I want you to select the user's favorite item in the list of candidate items following: {data["candidate_item_title"]}. Your response should be a item's title in the list of candidate items.
"""
    return {"input":prompt, 'output': data['output'], "user_id":data['user_id'], "candidate_item":data["candidate_item"], "task_type":"sequential", "history_items_id":data["history_items_id"]}

def generate_ctr_prompt(data):

    prompt = f"""The user's preferences are encoded in <unk> , additionally, he has recently interacted the following items he likes {data["history_list"]}.
    Based on the user's preferences and items he liked, I want you to determine if the user will like the item whose title is: "{data["candidate_item_title"]} and its encoded feature is <unk>.". Your response should be a simple "Yes" or "No" without any explanations.
"""
    return {"input":prompt, 'output': data['output'], "user_id":data['user_id'], "candidate_item":data["candidate_item"], "task_type":"ctr", "history_items_id":data["history_items_id"]}

def generate_rating_prompt(data):

    prompt = f"""The user whose preferences is <unk>  rated the following items he has interacted: {data["history_list"]}. 
    Based on the user's rating, now given your item: {data["candidate_item_title"]} with encoded feature is <unk>, you need to predict how will user rate for this item.Your response should be a number between 1-5 where 1 being lowest and 5 being highest.
"""
    return {"input":prompt, "output":data["output"], "user_id":data["user_id"], "candidate_item":data["candidate_item"], "task_type":"rating", "history_items_id":data["history_items_id"]}

def generate_exp_prompt(data):

    ### rating --> review 
    prompt = f"""Now, a user whose feature is <unk>'s comment on item {data["candidate_item_title"]} whose encoded feature is <unk>, the comment content is as follows:
    {data["history_list"]}. Based on user information and review content, how do you think the user would rate the item? Your response should be a number between 1-5 where 1 being lowest and 5 being highest.
"""
    return {"input":prompt, "output":data["output"], "user_id":data["user_id"], "candidate_item":data["candidate_item"], "task_type":"exp", "history_items_id":data["history_items_id"]}

def data_process(user_sequeneces_title:str, user_sequences_id:str, user_sequences_rating:str, user_sequences_review:str):

    user_title_sequences = load_json(user_sequeneces_title)
    user_id_sequences = load_json(user_sequences_id)
    user_sequences_rating = load_json(user_sequences_rating)
    user_sequences_review = load_json(user_sequences_review)
    id_2_title = load_json("../data/books_v2/id_2_title.json")
    all_item_id = set()
    for user in user_title_sequences:
        item_ids = user_id_sequences[user][1:]
        item_titles = user_title_sequences[user]
       
        assert len(item_ids) == len(item_titles)

        for id in item_ids:
            
            all_item_id.add(id)

    return id_2_title, all_item_id, user_title_sequences, user_id_sequences, user_sequences_rating, user_sequences_review

def generate_data(id_2_title, all_item_id, user_title_sequences,user_id_sequences, user_sequences_rating, user_sequences_review):

    ctr_train_data = []
    ctr_test_data = []
    sequential_train_data = []
    sequential_test_data = []
    rating_train_data = []
    rating_test_data = []
    exp_train_data = []
    exp_test_data = []
    users_recode= load_json('../data/books_v2/cold_users.json')
    cold_users = users_recode["cold_users_id"]
    print("cold_users:", len(cold_users))
    
    # user_list = list(user_id_sequences.keys())
    
    # random.shuffle(user_list)
    # train_user_ratio = 1
    # train_user_id_sequences = user_list[:int(train_user_ratio * len(user_list))]    
    
    for user in tqdm(user_id_sequences,desc="generate train data"):
        if int(user) in cold_users : continue
        user_id = (user_id_sequences[user][0])
        former_info = ""
        extra_info = ""
        for i in range(2):
            #### ctr预测任务数据集
            tmp = {}
            tmp["user_id"] = user_id
            try:
                start_pos = random.randint(0,3)
                end_pos = random.randint(start_pos+4, len(user_id_sequences[user][1:-2]))
            except:
                start_pos = 0
                end_pos = -2
            # item_list = user_id_sequences[user][1:15][:-1]    ### 最后一个商品是测试集
            # sample_number = random.randint(4,len(item_list)-3)
            try:
                history_list , candidate_item_id = user_id_sequences[user][1:][start_pos:end_pos][:10][:-1], user_id_sequences[user][1:][start_pos:end_pos][:10][-1]   ### 确保历史长度不超过10
            except:
                print(user_id_sequences[user][1:][start_pos:end_pos][:10])
                raise
            if random.random() < 0.5:
                tmp["output"] = "Yes."
            else:
                candidate_item_id = random.sample(all_item_id,1)[0]
                tmp["output"] = "No."
            
            tmp["history_list"] = [former_info + id_2_title[str(item)] + extra_info for item in history_list]
            tmp["history_items_id"] = history_list
            tmp["candidate_item_title"] = id_2_title[str(candidate_item_id)]
            tmp["candidate_item"] = [candidate_item_id] + [0] * 9
            ctr_train_data.append(tmp)
            
            #### 序列推荐数据集
            if i >= 0:
                tmp_seq = deepcopy(tmp)
                try:
                    start_pos = random.randint(0,3)
                    end_pos = random.randint(start_pos+4, len(user_id_sequences[user][1:-2]))
                except:
                    start_pos = 0
                    end_pos = -2
                history_list_sequences = user_id_sequences[user][1:][start_pos:end_pos][:10]   ### 确保历史长度不超过10
                history_list , target_item = history_list_sequences[:-1], history_list_sequences[-1]        
                tmp_seq["history_items_id"] = history_list
                tmp_seq["history_list"] = [former_info + id_2_title[str(item)] + extra_info for item in history_list]
                candidate_id_list = random.sample(all_item_id, 9)
                candidate_id_list += [target_item]

                random.shuffle(candidate_id_list)

                candidate_title_list = [ "item title: " + id_2_title[str(item)] + ", and its encoded feature is <unk>" for item in candidate_id_list]
                tmp_seq["candidate_item_title"] = candidate_title_list
                tmp_seq["candidate_item"] = candidate_id_list
                tmp_seq["output"] = id_2_title[str(target_item)]
                sequential_train_data.append(tmp_seq)

            #### 评分预测任务
            if i >= 1:
                tmp_rating = deepcopy(tmp)
                try:
                    start_pos = random.randint(0,3)
                    end_pos = random.randint(start_pos+4, len(user_id_sequences[user][1:-2]))
                except:
                    start_pos = 0
                    end_pos = -2
                target_item_id = user_id_sequences[user][1:][start_pos:end_pos][:10][-1]
                history_list_title = user_title_sequences[user][start_pos:end_pos][:10][:-1]
                history_list_rating = user_sequences_rating[user][start_pos:end_pos][:10][:-1]
                history_list = [former_info + item_title + extra_info + ", rating: " + str(item_rate) for item_title, item_rate in zip(history_list_title,history_list_rating)]
                tmp_rating["history_list"]=history_list
                tmp_rating["candidate_item"] = [target_item_id] + [0] * 9
                tmp_rating["history_items_id"] = user_id_sequences[user][1:][start_pos:end_pos][:10][:-1]
                tmp_rating["candidate_item_title"] = id_2_title[str(target_item_id)]
                tmp_rating["output"] = user_sequences_rating[user][start_pos:end_pos][:10][-1]
                rating_train_data.append(tmp_rating)
            ### 解释推荐
            if i>=1:
                tmp_exp = deepcopy(tmp)
                item_index = random.randint(0, len(user_id_sequences[user][1:])-3)
                rating = user_sequences_rating[user][item_index]
                item_title = user_title_sequences[user][item_index]
                review = " ".join(user_sequences_review[user][item_index].split(" ")[:100])
                tmp_exp["history_list"] = review
                tmp_exp["candidate_item_title"] = item_title
                tmp_exp["candidate_item"] = [user_id_sequences[user][1:][item_index]] + [0] * 9
                tmp_exp["history_items_id"] = [user_id_sequences[user][1:][item_index]]
                tmp_exp["output"] = rating
                exp_train_data.append(tmp_exp)

    
    for user in tqdm(user_id_sequences,desc="generate test data"): 
        
        test_tmp_ctr, test_tmp_seq, test_tmp_rating, test_tmp_exp = {}, {}, {}, {}
        test_ctr_each_user_data = []
        dic1 = {1:test_ctr_each_user_data, 2:test_tmp_seq, 3:test_tmp_rating, 4:test_tmp_exp}
        dic2 = {1:ctr_test_data, 2:sequential_test_data, 3:rating_test_data, 4:exp_test_data}
        flag = 0
        
        user_id = user_id_sequences[user][0]
        test_tmp_ctr["user_id"] = user_id
        test_history_length = random.randint(7,11)
        history_list = user_id_sequences[user][1:][-test_history_length:-1]
        test_rating_sequences = user_sequences_rating[user][-test_history_length:-1]
        test_title_sequences = user_title_sequences[user][-test_history_length:-1]
        test_tmp_ctr["history_items_id"] = history_list
        test_tmp_ctr["history_list"] =  [former_info + id_2_title[str(item)] + extra_info for item in history_list]
        for i in range(4):
            test_tmp_ctr_copy = deepcopy(test_tmp_ctr)
            if random.random() < 0.5:
                if flag == 1:continue
                candidate_item_id = user_id_sequences[user][-1]
                test_tmp_ctr_copy["output"] = "Yes."
                flag = 1
            else:
                candidate_item_id = random.sample(all_item_id,1)[0]
                test_tmp_ctr_copy["output"] = "No."
            
            test_tmp_ctr_copy["candidate_item"] = [candidate_item_id] + [0] * 9
            test_tmp_ctr_copy["candidate_item_title"] = id_2_title[str(candidate_item_id)]

            test_ctr_each_user_data.append(test_tmp_ctr_copy)
        # ctr_test_data.append(test_tmp_ctr)
        
        
        
        test_tmp_seq["user_id"] = user_id
        test_tmp_seq["history_list"] = [former_info + id_2_title[str(item)] + extra_info for item in history_list]
        test_tmp_seq["history_items_id"] = history_list
        candidate_id_list = random.sample(all_item_id, 9)
        candidate_id_list += [user_id_sequences[user][-1]]
        random.shuffle(candidate_id_list)
        candidate_title_list = ["item title: " + id_2_title[str(item)] + ", and its encoded feature is <unk>"  for item in candidate_id_list]
        test_tmp_seq["candidate_item_title"] = candidate_title_list
        test_tmp_seq["candidate_item"] = candidate_id_list
        test_tmp_seq["output"] = id_2_title[str(user_id_sequences[user][-1])]
        # sequential_test_data.append(test_tmp_seq)
        

       
        test_tmp_rating["user_id"] = user_id
        test_tmp_rating["history_list"] = [former_info + item_title + extra_info + ", rating: " + str(item_rate) for item_title, item_rate in zip(test_title_sequences, test_rating_sequences)]
        test_tmp_rating["candidate_item_title"] = user_title_sequences[user][-1]
        test_tmp_rating["history_items_id"] = history_list
        test_tmp_rating["candidate_item"] = [user_id_sequences[user][-1]] + [0] * 9
        test_tmp_rating["output"] = user_sequences_rating[user][-1]
        # rating_test_data.append(test_tmp_rating)

        
        test_tmp_exp["user_id"] = user_id
        test_tmp_exp["history_list"] = " ".join(user_sequences_review[user][-1].split(" ")[:100]) ### limit 100 words
        test_tmp_exp["candidate_item_title"] = user_title_sequences[user][-1]
        test_tmp_exp["candidate_item"] = [user_id_sequences[user][-1]] + [0] * 9
        test_tmp_exp["history_items_id"] = [user_id_sequences[user][-1]]
        test_tmp_exp["output"] = user_sequences_rating[user][-1]
        # exp_test_data.append(test_tmp_exp)
        seed = random.randint(1,4)
        if seed == 1:
            dic2[seed].extend(test_ctr_each_user_data)
        else:
            dic2[seed].append(dic1[seed])

    return ctr_train_data, ctr_test_data, sequential_train_data, sequential_test_data, rating_train_data, rating_test_data, exp_train_data, exp_test_data , cold_users
                        
def main():   
    id_2_title, all_item_id, user_title_sequences,user_id_sequences, user_rating_sequences, user_review_sequences = data_process('../data/books_v2/user_sequences_title.json', '../data/books_v2/user_sequences_id.json', '../data/books_v2/user_rating.json', '../data/books_v2/user_review.json')
    ctr_train_data, ctr_test_data ,sequential_train_data, sequential_test_data, rating_train_data, rating_test_data, exp_train_data, exp_test_data, cold_users = generate_data(id_2_title, all_item_id, user_title_sequences, user_id_sequences, user_rating_sequences, user_review_sequences)
    ctr_train = []
    ctr_test = []
    sequential_train = []
    sequential_test = []
    rating_train = []
    rating_test = []
    exp_train = []
    exp_test = []

    for data in (ctr_train_data):
        ctr_train.append(generate_ctr_prompt(data))
    for data in ctr_test_data:
        ctr_test.append(generate_ctr_prompt(data))
    random.shuffle(rating_train)
    print("ctr train data length: ", len(ctr_train))


    for data in sequential_train_data:
        sequential_train.append(generate_sequential_recommend_prompt(data))
    print("sequential train data length: ", len(sequential_train))
    for data in sequential_test_data:
        sequential_test.append(generate_sequential_recommend_prompt(data))


    for data in rating_train_data:
        rating_train.append(generate_rating_prompt(data))
    print("rating data length: ", len(rating_train))
    for data in rating_test_data:
        rating_test.append(generate_rating_prompt(data))

    for data in exp_train_data:
        exp_train.append(generate_exp_prompt(data))
    print("exp data length: ", len(exp_train))
    for data in exp_test_data:
        exp_test.append(generate_exp_prompt(data))
    print("total cold users: ", len(cold_users))
    warm_user_datas , cold_user_datas = [] , []
    for datas in tqdm([ctr_test, sequential_test, rating_test, exp_test]):
        for data in tqdm(datas):
            if data["user_id"] in cold_users:
                cold_user_datas.append(data)
            else:
                warm_user_datas.append(data)
    print(len(warm_user_datas))
    print(len(cold_user_datas))
    train_data = ctr_train + sequential_train + rating_train + exp_train
    random.shuffle(train_data)
    print("train data length: ", len(train_data))
    
    test_data = warm_user_datas + cold_user_datas
    print("test_data length: ", len(test_data))
    random.shuffle(test_data)
    print(test_data[100])
    # with jsonlines.open('../data/books_for_train/train.jsonl', 'w') as writer:
    #     writer.write_all(train_data)

    # # with jsonlines.open('../data/books_for_train/test_warm.jsonl', 'w') as writer:
    # #     writer.write_all(warm_user_datas)
    
    # # with jsonlines.open('../data/books_for_train/test_cold.jsonl', 'w') as writer:
    # #     writer.write_all(cold_user_datas)

    with jsonlines.open('../data/books_for_train/test.jsonl', 'w') as writer:
        writer.write_all(test_data)

if __name__ == "__main__":
    main()
