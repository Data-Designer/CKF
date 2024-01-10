from copy import deepcopy
import os
import torch
from datasets import load_dataset  
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, AdamW, get_linear_schedule_with_warmup, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq, GenerationConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, LoraModel, PeftConfig
from torch import nn, sigmoid
from typing import Optional, Union, Tuple, List , Callable
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.utils.data import Dataset, DataLoader
import random
import jsonlines
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import math
from torch.utils.data.distributed import DistributedSampler
from personalMapping import  AttentionNet, MetaNet
import json

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.autograd.set_detect_anomaly(True)
model_name = "/data1/model_repository/llama-2-7b-chat-hf"
lora_r = 16
lora_alpha = 64
lora_dropout = 0.1
max_grad_norm = 0.3
weight_decay = 0.001
warmup_ratio = 0.05
max_seq_length = 2048
device = "cuda:0"
accumulation_steps = 1
EPOCHES = 3
LR = 1e-4
eps = 1e-6

base_dir = ""  
user_embedding = torch.load(f'{base_dir}/model_weight/books_v2/books_MF_user_embedding.pkl')
user_embedding = nn.Embedding.from_pretrained(user_embedding)

item_embedding = torch.load(f'{base_dir}/model_weight/books_v2/books_MF_item_embedding.pkl')
item_embedding = nn.Embedding.from_pretrained(item_embedding.weight)

#embeddings = torch.load('books_saved_embeddings.pth')
#user_embedding = nn.Embedding.from_pretrained(embeddings["user_embedding"])
#item_embedding = nn.Embedding.from_pretrained(embeddings["item_embedding"])
tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True, add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = False
tokenizer.padding_side = "left"

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

    def __init__(self, tokenizer, mode="train", data_type=None, device=None):

        self.tokenizer = tokenizer
        self.mode = mode
        self.data_type = data_type
        self.device = device

    def mapping_funcation(self, input_list, concate_list,):
        
        input_text_start_index = input_list.index(1)
        input_length = len(input_list[input_text_start_index:])
        for i in range(len(concate_list)):
            if concate_list[i:i+input_length] == input_list[input_text_start_index:]:
                return i + input_length
        

    def __call__(self, batch):
        
        task_2_id = {"ctr":1 , "sequential":2, "rating":3, "exp":4}
        input_texts = [item['input'] for item in batch]
        concate_texts = [item["input"]+str(item["output"])+'</s>' for item in batch]
        
        user_id = [item['user_id'] for item in batch]
        
        candidate_item = [ item['candidate_item']  for item in batch]
        input_text_ids = self.tokenizer(input_texts, padding="longest", truncation=True, max_length = max_seq_length,  return_tensors=None)
        input_concate_output_ids = self.tokenizer(concate_texts, padding="longest", truncation=True, max_length = max_seq_length, return_tensors=None)
        labels_start_index  = [self.mapping_funcation(input_text_ids["input_ids"][i], input_concate_output_ids["input_ids"][i]) for i in range(len(batch))]
        task_id = [task_2_id[item['task_type']] for item in batch]
        labels = [[-100] * labels_start_index[i] + input_concate_output_ids["input_ids"][i][j:] for i, j in zip(range(len(batch)), labels_start_index)]
        
        output_dict = {}
        output_dict["data_type"] = torch.tensor(1) if self.data_type == "title" else torch.tensor(0)
        output_dict["input_ids"] = torch.tensor(input_concate_output_ids["input_ids"]) if self.mode=="train" else torch.tensor(input_text_ids["input_ids"])
        output_dict["attention_mask"] = torch.tensor(input_concate_output_ids["attention_mask"]) if self.mode=="train" else torch.tensor(input_text_ids["attention_mask"])
        output_dict["labels"] = torch.tensor(labels) if self.mode=="train" else torch.tensor(input_text_ids["input_ids"])
        output_dict["user_id"] = torch.tensor(user_id)
        output_dict["candidate_item"] = torch.tensor(candidate_item)
        output_dict["task_id"] = torch.tensor(task_id)
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
                new_user_embeds = self.personal_mapping(batch["user_id"][i]).clone()
                mapping_user_embeds = self.user_mapping(new_user_embeds).half()
                new_item_embeds = self.item_mapping(self.item_embedding[batch["candidate_item"][i]]).half()
                input_embeds_new[i][unk_id_index[0]] = mapping_user_embeds
                input_embeds_new[i][unk_id_index[1:].view(1,-1)] = new_item_embeds
            else:
                new_user_embeds = self.personal_mapping(batch["user_id"][i]).clone()
                mapping_user_embeds = self.user_mapping(new_user_embeds).half()
                new_item_embeds = self.item_mapping(self.item_embedding[batch["candidate_item"][i][0]]).half()
                input_embeds_new[i][unk_id_index[0]] = mapping_user_embeds
                input_embeds_new[i][unk_id_index[1]] = new_item_embeds
        output2 = self.model(inputs_embeds=input_embeds_new, attention_mask = cf_attention_mask,  labels=cf_labels)
        return  input_embeds_new,  output2["loss"], output2["logits"]
    
def generate_optimizer_scheduler(parameters, data_size):
    batch_per_epoch = data_size 
    t_total = batch_per_epoch // accumulation_steps * EPOCHES
    warmup_iters = int(t_total * warmup_ratio)

    print("Batch per epoch: %d" % batch_per_epoch)
    print("Total Iters: %d" % t_total)
    print('Warmup ratio:', warmup_ratio)
    print("Warm up Iters: %d" % warmup_iters)

    optim = AdamW(filter(lambda p: p.requires_grad, parameters),   ### LR 1e-4
                  lr=LR, eps=eps, weight_decay=weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optim, warmup_iters, t_total)

    return optim, lr_scheduler

def get_dataloader(title_data_file, cf_data_file, batch_size=8, f_filter=None):

    title_dataset = Construct_dataset(data_file=title_data_file, filter_f=f_filter)
    cf_dataset = Construct_dataset(data_file=cf_data_file, filter_f=f_filter)

    assert len(title_dataset) == len(cf_dataset)


    collator1 = Collator(tokenizer=tokenizer, mode="train", data_type="title")
    collator2 = Collator(tokenizer=tokenizer, mode="train", data_type="cf")
    title_dataloader = DataLoader(title_dataset, batch_size=batch_size, collate_fn=collator1, shuffle=False, drop_last=True,)
    cf_dataloader = DataLoader(cf_dataset, batch_size=batch_size, collate_fn=collator2, shuffle=False, drop_last=True, )
    assert len(title_dataloader) == len(cf_dataloader)
    return title_dataloader, cf_dataloader

def sigmoid_func(z , bias):
    try: 
        return 1 / (1 + math.exp(8 * (z - bias)/bias))
    except:
        return 0

def print_trainable_parameters(net):
    trainable_params = 0
    all_param = 0
    for _, param in net.named_parameters():
        num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )


def get_optim(net , dataloader, task_type):
    for name , param in net.named_parameters():
        if "adapter" in name or "embedding" in name:
            param.requires_grad = False

    optim2, lr_scheduler2 = generate_optimizer_scheduler(net.parameters(), len(dataloader))

    for name , param in net.named_parameters():
        if f"{task_type}" in name or "share" in name:
            param.requires_grad = True

    optim1, lr_scheduler1 = generate_optimizer_scheduler(net.parameters(), len(dataloader))

    for name , param in net.named_parameters():
        print(name , param.requires_grad)

    print_trainable_parameters(net)
    return optim1, lr_scheduler1 , optim2, lr_scheduler2

def main():
    llama_model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16, device_map=device)
    llama_model = prepare_model_for_int8_training(llama_model)
    lora_config1 = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "v_proj", "o_proj"]
    )

    lora_config2 = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj"]
    )

    lora_config3 = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj"]
    )

    lora_config4 = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj"]
    )

    lora_config5 = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj"]
    )

    model1= get_peft_model(llama_model, lora_config1, adapter_name="adapter_share")
    model1.print_trainable_parameters()

    model1.add_adapter(adapter_name="adapter_ctr", peft_config=lora_config2)
    model1.add_adapter(adapter_name="adapter_sequential", peft_config=lora_config3)
    model1.add_adapter(adapter_name="adapter_rating", peft_config=lora_config4)
    model1.add_adapter(adapter_name="adapter_exp", peft_config=lora_config5)
    
    user_sequences_file = f'{base_dir}/data/books_v2/user_sequences_id.json'
    user_sequences = json.load(open(user_sequences_file, 'r'))
    net = Net(model1, user_embedding=user_embedding , item_embedding=item_embedding, user_sequences=user_sequences).to(device)

   
    train_title_file = f'{base_dir}/data/books_for_train/all_title_train.jsonl'
    train_cf_data_file = f'{base_dir}/data/books_for_train/train.jsonl'
    title_dataloader1, cf_dataloader1 = get_dataloader(title_data_file=train_title_file, cf_data_file=train_cf_data_file, batch_size=16, f_filter=lambda x : x["task_type"]=="ctr")
    title_dataloader2, cf_dataloader2 = get_dataloader(title_data_file=train_title_file, cf_data_file=train_cf_data_file, batch_size=16, f_filter=lambda x : x["task_type"]=="sequential")
    title_dataloader3, cf_dataloader3 = get_dataloader(title_data_file=train_title_file, cf_data_file=train_cf_data_file, batch_size=8, f_filter=lambda x : x["task_type"]=="rating")
    title_dataloader4, cf_dataloader4 = get_dataloader(title_data_file=train_title_file, cf_data_file=train_cf_data_file, batch_size=8, f_filter=lambda x :x["task_type"]=="exp")
    assert len(title_dataloader1) == len(title_dataloader2) == len(title_dataloader3) 
    total_steps = EPOCHES * len(title_dataloader1)
    sigmoid_bias1 = total_steps // 2 
    
    loss_fn = nn.CrossEntropyLoss()
    ctr_optim1, ctr_lr_scheduler1, ctr_optim2, ctr_lr_scheduler2 = get_optim(net, cf_dataloader1, task_type="ctr")
    seq_optim1, seq_lr_scheduler1, seq_optim2, seq_lr_scheduler2 = get_optim(net, cf_dataloader2, task_type="sequential")
    rating_optim1, rating_lr_scheduler1,rating_optim2, rating_lr_scheduler2 = get_optim(net,cf_dataloader1, task_type="rating")
    exp_optim1, exp_lr_scheduler1, exp_optim2, exp_lr_scheduler2 = get_optim(net,cf_dataloader2, task_type="exp")
    optim1 = {1: ctr_optim1, 2: seq_optim1, 3: rating_optim1, 4: exp_optim1}
    lr_scheduler1 = {1: ctr_lr_scheduler1, 2: seq_lr_scheduler1, 3: rating_lr_scheduler1, 4: exp_lr_scheduler1}
    optim2 = {1: ctr_optim2, 2: seq_optim2, 3: rating_optim2, 4: exp_optim2}
    lr_scheduler2 = {1: ctr_lr_scheduler2, 2: seq_lr_scheduler2, 3: rating_lr_scheduler2, 4: exp_lr_scheduler2} 

    for epoch in range(EPOCHES):
    
        for j , (batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8) in tqdm(enumerate(zip(title_dataloader1, cf_dataloader1, title_dataloader2,cf_dataloader2,
                                                       title_dataloader3, cf_dataloader3, title_dataloader4, cf_dataloader4)), total=len(cf_dataloader1), desc="train multi lora task"):
            net.train()
            loss_dict = {}
            batch_dict = {1:[batch1, batch2], 2: [batch3, batch4], 3: [batch5, batch6], 4: [batch7, batch8]}
            loss_weight = sigmoid_func(epoch*len(title_dataloader1)+j+1, sigmoid_bias1) 
            for i in range(1,5):
                loss_dict[i] = loss_dict.get(i,[]) + [net(batch_dict[i][0]) * loss_weight]
                _ , loss2, _ = net(batch_dict[i][1])
                loss_dict[i] = loss_dict.get(i, []) + [loss2 * (1 - loss_weight)]
            # print(loss_dict)
            # for i in range(1,5):
                
                loss_dict[i][0].backward()
                optim1[i].step()
                lr_scheduler1[i].step()
                optim1[i].zero_grad()

                loss_dict[i][1].backward()
                optim2[i].step()
                lr_scheduler2[i].step()
                optim2[i].zero_grad()
            
            if (j+1) % 100 == 0:
                print(loss_dict)
        
        net.model.save_pretrained(save_directory=f"../model_weight/books_mf_llama2/epoch_{epoch}")

        print(f"------------------------------ {epoch+1} has finished------------------------------")
        torch.save(net.user_mapping, "../model_weight/books_user_mapping_layer_mf_llama2.pkl")
        torch.save(net.item_mapping, "../model_weight/books_item_mapping_layer_mf_llama2.pkl")

if __name__ == "__main__":
    main()
