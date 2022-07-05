import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm.auto import tqdm


import streamlit as st


SAVED_MODEL_DIR = "./SAVED_MODEL_DIR"
max_target_length = 64

class KeicoDataset(Dataset):
    def __init__(self, tokenizer, df_input, df_target):        
        self.df_input = df_input
        self.df_target = df_target
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()
    
    def __len__(self):
        return len(self.inputs)
    
    def _build(self):
        for i in range(len(self.df_input)):
            if type(self.df_input[i]) == str and type(self.df_target[i]) == str:
                tokenized_inputs = tokenizer.batch_encode_plus(
                    [self.df_input[i]], max_length=64, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                tokenized_targets = tokenizer.batch_encode_plus(
                    [self.df_target[i]], max_length=64, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

tokenizer = T5Tokenizer.from_pretrained(SAVED_MODEL_DIR, is_fast=True)
trained_model = T5ForConditionalGeneration.from_pretrained(SAVED_MODEL_DIR)

# GPUの利用有無
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    trained_model.cuda()


trained_model.eval()


#标题
st.header("敬語自動変換システム")

#输入框
Mysentence = [st.text_input("変換したい文を64文字以内で入力してください。")]
test_dataset = KeicoDataset(tokenizer, Mysentence, [""])
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=0)


#点击提交按钮
if st.button("Submit"):
    #引入训练好的模型

    inputs = []
    outputs = []

    for batch in tqdm(test_loader):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']
        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        output = trained_model.generate(input_ids=input_ids, 
            attention_mask=input_mask, 
            max_length=max_target_length,
            repetition_penalty=10.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
            )

        output_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False) 
                    for ids in output]
        outputs.extend(output_text)

    


    #返回预测的值
    st.text(f"{outputs[0]}")
