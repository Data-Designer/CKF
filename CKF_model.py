import torch
from datasets import load_dataset  
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, AdamW, get_linear_schedule_with_warmup, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq, GenerationConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, LoraModel, PeftConfig
from torch import nn, sigmoid
from personalMapping import AttentionNet, MetaNet

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
        ### input is all text without CF signal , only optimize Lora
        if batch["data_type"] == 1:
            all_title_input_embeds = self.model.get_input_embeddings()(batch["input_ids"])
            all_title_attention_mask = batch["attention_mask"]
            all_title_labels = batch["labels"]
            output1 = self.model(inputs_embeds=all_title_input_embeds, attention_mask=all_title_attention_mask, labels=all_title_labels)
            return output1["loss"]
        
        ### input data contains CF signal , only optimize Mapping layer 
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

