import os
import torch
from datasets import load_dataset  
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, AdamW, get_linear_schedule_with_warmup, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq, GenerationConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, LoraModel, PeftConfig, PeftModel
from torch import ne, nn
from typing import Optional, Union, Tuple, List , Callable
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.utils.data import Dataset, DataLoader
import random
import jsonlines
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
import math 
from personalMapping import  AttentionNet, MetaNet
import json

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#model_name = "/data1/model_repository/llama-2-7b-chat-hf"
model_name = "/data1/llama-7b-hf"
lora_r = 32
lora_alpha = 64
lora_dropout = 0.1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "adamw_torch"
warmup_ratio = 0.05
group_by_length = True
save_steps = 25
logging_steps = 10
max_seq_length = 2048
packing = False
device = "cuda:0"
base_dir = ""
user_embedding = torch.load(f'{base_dir}/model_weight/books_v2/books_MF_user_embedding.pkl')
user_embedding = nn.Embedding.from_pretrained(user_embedding)
item_embedding = torch.load(f'{base_dir}/model_weight/books_v2/books_MF_item_embedding.pkl')
item_embedding = nn.Embedding.from_pretrained(item_embedding.weight)



tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True, add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = False
tokenizer.padding_side = "left"
print(tokenizer.convert_ids_to_tokens(8241))

f = open('../data/books_for_train/cold_users.json','r')
content = f.read()
user_recodes = json.loads(content)
cold_users = user_recodes["cold_users_id"]

class Construct_dataset(Dataset):

    def __init__(self, data_file:str, filter_f=None):

        self.datas = list(jsonlines.open(data_file, 'r'))
        
        self.total_datas = []
        if filter_f:
            for data in filter(filter_f, self.datas):
                self.total_datas.append(data)
        else:
            self.total_datas = self.datas
            
        
        random.shuffle(self.total_datas)
    
    def __getitem__(self, index):
        return self.total_datas[index]    
    
    def __len__(self):
        return len(self.total_datas)
    

class Collator(object):

    def __init__(self, tokenizer, mode="train"):

        self.tokenizer = tokenizer
        self.mode = mode
      

    def mapping_funcation(self, input_list, concate_list,):
        
        input_text_start_index = input_list.index(1)
        input_length = len(input_list[input_text_start_index:])
        for i in range(len(concate_list)):
            if concate_list[i:i+input_length] == input_list[input_text_start_index:]:
                return i + input_length
        
            

    def __call__(self, batch):

        input_texts = [item['input'] for item in batch]
        concate_texts = [item["input"]+str(item["output"])+'</s>' for item in batch]
        
        user_id = [item['user_id'] for item in batch]
        candidate_item = [item['candidate_item'] for item in batch]
      
        input_text_ids = self.tokenizer(input_texts, padding="longest", truncation=True, max_length = max_seq_length,  return_tensors=None)
        input_concate_output_ids = self.tokenizer(concate_texts, padding="longest", truncation=True, max_length = max_seq_length, return_tensors=None)
        labels_start_index  = [self.mapping_funcation(input_text_ids["input_ids"][i], input_concate_output_ids["input_ids"][i]) for i in range(len(batch))]
        
       
        labels = [[-100] * labels_start_index[i] + input_concate_output_ids["input_ids"][i][j:] for i, j in zip(range(len(batch)), labels_start_index)]
        
        output_dict = {}
        output_dict["input_ids"] = torch.tensor(input_concate_output_ids["input_ids"]) if self.mode=="train" else torch.tensor(input_text_ids["input_ids"]).to(device)
        output_dict["attention_mask"] = torch.tensor(input_concate_output_ids["attention_mask"]) if self.mode=="train" else torch.tensor(input_text_ids["attention_mask"]).to(device)
        output_dict["labels"] = torch.tensor(labels) if self.mode=="train" else torch.tensor(input_text_ids["input_ids"]).to(device)
        output_dict["user_id"] = torch.tensor(user_id).to(device)
        output_dict["data_type"] = torch.tensor(0)
        output_dict["candidate_item"] = torch.tensor(candidate_item).to(device)
        if self.mode != "train":
            output_dict["true_label"] = [item["output"] for item in batch]

        return output_dict

class Net(nn.Module):

    def __init__(self, llama_model ,user_embedding, item_embedding, user_sequences):

        super().__init__()
        
        self.model = llama_model
        self.user_embedding = user_embedding.weight
        self.item_embedding = item_embedding.weight
        # self.user_embedding.requires_grad = False
        # self.item_embedding.requires_grad = False
        self.user_mapping = nn.Linear(8,4096)
        self.item_mapping = nn.Linear(8,4096)
        self.attention_net = AttentionNet(input_dim=8, hidden_states=[32, 64, 8])

        self.Meta_net = MetaNet(input_dim=8, layers_dim=[32, 64])
        self.user_sequences = user_sequences

    def personal_mapping(self, user):
        
        sequences = self.user_sequences[str(user.item())]
        x = torch.LongTensor(sequences).unsqueeze(0)
        user_history = x[:,1:-1]
        user_history_embedding = self.item_embedding[user_history]

        # history_embedding = self.attention_net(user_history_embedding)
        pu = torch.mean(user_history_embedding, dim=1)
        user_history_embedding_ = self.Meta_net(pu)
        # pu = torch.mean(user_history_embedding_.clone(), dim=1)

        wu = user_history_embedding_.reshape(-1,8,8)
        # wu = self.Meta_net(pu).reshape(-1, 8, 8)
        user_embedding = self.user_embedding[x[:,0]]
        personal_mapping_embedding = torch.matmul(user_embedding.clone().unsqueeze(1), wu).clone().squeeze(1)
        return personal_mapping_embedding

    def print_trainable_parameters(self):
       self.model.print_trainable_parameters()

    def forward(self, batch):
        #### batch1 : all title data , batch: title with cf infomation
        batch_size = len(batch["input_ids"])
        ### 输入为全title , 只优化Lora模块
        if batch["data_type"] == 1:
            all_title_input_embeds = self.model.get_input_embeddings()(batch["input_ids"])
            all_title_attention_mask = batch["attention_mask"]
            all_title_labels = batch["labels"]
            output1 = self.model(inputs_embeds=all_title_input_embeds, attention_mask=all_title_attention_mask, labels=all_title_labels)
            return output1["loss"]
        
        ### 输入包含 CF 信号，只优化mapping
        input_embeds = self.model.get_input_embeddings()(batch["input_ids"])
        cf_attention_mask = batch["attention_mask"]
        cf_labels = batch["labels"]
        for i in range(batch_size):
            unk_id_index = torch.nonzero(batch["input_ids"][i] == 0)
            input_embeds_new = input_embeds.clone().half()
           
            if len(unk_id_index) != 2:
                try:
                    new_user_embeds = self.personal_mapping(batch["user_id"][i]).clone()
                except:
                    new_user_embeds = self.user_embedding[batch["user_id"][i]].clone()
                mapping_user_embeds = self.user_mapping(new_user_embeds).half()
                new_item_embeds = self.item_mapping(self.item_embedding[batch["candidate_item"][i]]).half()
                input_embeds_new[i][unk_id_index[0]] = mapping_user_embeds
                input_embeds_new[i][unk_id_index[1:].view(1,-1)] = new_item_embeds
            else:
                try:
                    new_user_embeds = self.personal_mapping(batch["user_id"][i]).clone()
                except:
                    new_user_embeds = self.user_embedding[batch["user_id"][i]].clone()
                mapping_user_embeds = self.user_mapping(new_user_embeds).half()
                new_item_embeds = self.item_mapping(self.item_embedding[batch["candidate_item"][i][0]]).half()
                input_embeds_new[i][unk_id_index[0]] = mapping_user_embeds
                input_embeds_new[i][unk_id_index[1]] = new_item_embeds
        output2 = self.model(inputs_embeds=input_embeds_new, attention_mask = cf_attention_mask,  labels=cf_labels)
        return  input_embeds_new,  output2["loss"], output2["logits"]


def evaulate_ctr(net, test_dataloader_2):
    warm_user_predict = []
    warm_groundtruth = []
    cold_user_predict = []
    cold_groundtruth = []

    user_predict = {}
    user_labels = {}

    logits_list = []
    gold_list = []
    net.eval()
    generation_config = GenerationConfig(
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id,

)
    count = 0
    with torch.no_grad():
   
       
        print("-------------------------start eval ctr--------------------------")
        for batch2 in tqdm( test_dataloader_2, total=len(test_dataloader_2)):


            input_embeds , _ , _= net(batch2)
           
            
            gold_list += [int(batch2["true_label"][0] == "Yes.")]
            # print(input_embeds.shape)
            output = net.model.generate(inputs_embeds=input_embeds, pad_token_id=tokenizer.eos_token_id, 
                                        max_new_tokens = 128, return_dict_in_generate=True,
                                        output_scores=True,
                                        generation_config = generation_config)
            s = output.sequences
            scores = output.scores[0].softmax(dim=-1)
            logits = torch.tensor(scores[:,[8241, 3782]], dtype=torch.float32).softmax(dim=-1)
            # print(logits, batch2["true_label"][0])
            logits_list += [logits.tolist()[0][0]]

            user_id = batch2["user_id"][0].cpu().item()
            user_predict[user_id] = user_predict.get(user_id, [])
            user_labels[user_id] = user_labels.get(user_id, [])
            user_labels[user_id] += [int(batch2["true_label"][0] == "Yes.")]      
            user_predict[user_id] += [logits.tolist()[0][0]]

            if batch2["user_id"][0] not in cold_users:
                warm_user_predict += [logits.tolist()[0][0]]
                warm_groundtruth += [int(batch2["true_label"][0] == "Yes.")]
            else:
                cold_user_predict += [logits.tolist()[0][0]]
                cold_groundtruth += [int(batch2["true_label"][0] == "Yes.")]



    print("ctr auc: ", roc_auc_score(gold_list, logits_list))
    print("warm user auc: ", roc_auc_score(warm_groundtruth, warm_user_predict))
    print("cold user auc: ", roc_auc_score(cold_groundtruth, cold_user_predict))


    user_auc = 0
    count = 0
    for user in tqdm(user_labels):
        if len(set(user_labels[user])) == 1:
            continue 
        each_user_auc = roc_auc_score(user_labels[user], user_predict[user])
        count += 1
        user_auc += each_user_auc
    print("ctr UAUC: ", user_auc / (count))

def evaulate_sequential(net, test_dataloader):
    warm_user_hit = 0
    warm_user_total = 0
    cold_user_hit = 0
    cold_user_total = 0

    sequential_hit = 0
    sequential_total = 0
    net.eval()
    generation_config = GenerationConfig(
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id,
    do_sample = True,
    top_k = 10,
    top_p=0.95,
    early_stopping = True,
    num_return_sequences = 1,
)
    count = 0
    with torch.no_grad():
        print("-------------------------start eval sequential--------------------------")
        for batch in tqdm(test_dataloader):
            
            input_embeds , _ , _ = net(batch)
            # print(input_embeds.shape)
            output = net.model.generate(inputs_embeds=input_embeds, pad_token_id=tokenizer.eos_token_id, 
                                        max_new_tokens = 128, 
                                        generation_config = generation_config)
            res = (tokenizer.batch_decode(output, skip_special_tokens=True))

            if batch["user_id"][0] not in cold_users:
                warm_user_total += 1
                if (res[0]) == batch["true_label"][0]:
                    warm_user_hit += 1
            else:
                cold_user_total += 1
                if (res[0]) == batch["true_label"][0]:
                    cold_user_hit += 1
            sequential_total += 1
            if (res[0]) == batch["true_label"][0]:
                    sequential_hit += 1
        print("warm user accuracy: ", warm_user_hit / warm_user_total)
        print("cold user accuracy: ", cold_user_hit / cold_user_total)
        print("sequential accuracy: ", sequential_hit / sequential_total)  
        print("-------------------------end eval sequential--------------------------")

def evaulate_rating(net, test_dataloader):
    warm_user_predict = []
    warm_groundtruth = []
    cold_user_predict = []
    cold_groundtruth = []
    all_predict = []
    all_groundtruth = []

    generation_config = GenerationConfig(
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id,
    do_sample = True,
    top_k = 10,
    top_p=0.95,
    early_stopping = True,
    num_return_sequences = 1,
)
    net.eval()
    count = 0
    with torch.no_grad():
        
        
        print("-------------------------start eval rating--------------------------")
        for batch in tqdm(test_dataloader):
            
            input_embeds , _ , _ = net(batch)
        # print(input_embeds.shape)
            output = net.model.generate(inputs_embeds=input_embeds, pad_token_id=tokenizer.eos_token_id, 
                                        max_new_tokens = 128, 
                                        generation_config = generation_config)
            res = (tokenizer.batch_decode(output, skip_special_tokens=True))      

            if batch["user_id"][0] not in cold_users:
                
                try :
                    warm_user_predict += [float(res[0])]
                    warm_groundtruth += [float(batch["true_label"][0])]
                except:
                    print("output error1")
            else:
                try :
                    cold_user_predict += [float(res[0])]
                    cold_groundtruth += [float(batch["true_label"][0])]
                except:
                    print("output error2")
            try:
                all_predict += [float(res[0])]
                all_groundtruth += [float(batch["true_label"][0])]
            except:
                print("output error3")
        print("rating MAE", mean_absolute_error(all_groundtruth, all_predict))
        print("rating MSE", mean_squared_error(all_groundtruth, all_predict))
        print("warm user MAE", mean_absolute_error(warm_groundtruth, warm_user_predict))
        print("warm user MSE", mean_squared_error(warm_groundtruth, warm_user_predict))
        print("cold user MAE", mean_absolute_error(cold_groundtruth, cold_user_predict))
        print("cold user MSE", mean_squared_error(cold_groundtruth, cold_user_predict))
        print("-------------------------end eval rating--------------------------")


def main(task, epoch):
    user_sequences_file = f'{base_dir}/data/books_for_train/user_sequences_id.json'
    user_sequences = json.load(open(user_sequences_file, 'r'))
    pretrained_user_mapping = torch.load('../model_weight/books_user_mapping_layer_MF.pkl', map_location=device) 
    pretrained_item_mapping = torch.load('../model_weight/books_item_mapping_layer_MF.pkl', map_location=device) 
    test_dataset = f'{base_dir}/data/books_for_train/test.jsonl'
    cf_dataset = Construct_dataset(data_file=test_dataset,  filter_f=lambda x:x["task_type"]==f"{task}")
    collator = Collator(tokenizer=tokenizer, mode="test")
    test_dataloader = DataLoader(cf_dataset, batch_size=1, collate_fn=collator, shuffle=True)
    llama_casual = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(llama_casual, f'../model_weight/books_MF/epoch_{epoch}/adapter_share', adapter_name=f"adapter_share")
    model = PeftModel.from_pretrained(llama_casual, f'../model_weight/movie_din_llama2/epoch_{epoch}/adapter_{task}', adapter_name=f"adapter_{task}")
    test_model = model.merge_and_unload()
  
    test_model.user_mapping = pretrained_user_mapping
    test_model.item_mapping = pretrained_item_mapping
    eval_net = Net(test_model, user_embedding, item_embedding, user_sequences)
    eval_net.to(device)
    if task == "sequential":
        evaulate_sequential(eval_net ,  test_dataloader)
    elif task == "ctr":
        evaulate_ctr(eval_net ,  test_dataloader)
    else:
        evaulate_rating(eval_net ,  test_dataloader)

if __name__ == "__main__":
    for i in range(2):
        for task in ["ctr","sequential", "rating", "exp"]:
            main(task,i)
