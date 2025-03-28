import json
from tqdm import tqdm
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5').to(device='cuda:1')
model.eval()

def embedding(text):
    if isinstance(text, str):
        text = [text]
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda:1')
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    text_embeddings = model_output[0][:, 0]
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
    return text_embeddings[0].cpu().tolist()

#-------calculate the embedding of news -----------

# meso_news_filepath_list = ["../raw_data/merged_data_except_2020.json", 
#                            "../raw_data/data_2020.json"]
meso_news_filepath_list = [
                           "../raw_data/data_2020.json"]

for meso_news_filepath in meso_news_filepath_list:
    meso_news_data = json.load(open(meso_news_filepath, "r", encoding="utf-8"))

    meso_news_embeddings = []

    for i in tqdm(range(len(meso_news_data))):
        news_text = meso_news_data[i]["title"] + "\n" + meso_news_data[i]["body"]
        news_text_embedding = embedding(news_text)
        meso_news_embeddings.append(np.array(news_text_embedding))

    np.save(meso_news_filepath.replace(".json", "_embedding.npy"), np.array(meso_news_embeddings))

#-------calculate the embedding of policies definition -----------
policy_definition_filepath = "../raw_data/policy_definition.json"
with open(policy_definition_filepath, "r", encoding='utf-8') as f:
    policy_definition_data = json.load(f)
    
for i in tqdm(range(len(policy_definition_data))):
    policy_definition_data[i]["vector"] = embedding(policy_definition_data[i]["definition"])

data2 = {"rows": policy_definition_data}
policy_definition_filepath = "../raw_data/policy_definition_embedding.json"
with open(policy_definition_filepath, "w", encoding='utf-8') as f:
    json.dump(data2, f, indent=4, ensure_ascii=False)


#-------calculate the embedding of industry_definition -----------
industry_definition_embeddings = []
industry_definition_file = "../raw_data/industry_definition.xlsx"
industry_definition_df = pd.read_excel(industry_definition_file, header=None)
industries = {}
for i in tqdm(range(len(industry_definition_df))):
    industry = industry_definition_df.iloc[i, 0]
    industry_text = industry_definition_df.iloc[i, 1]
    industry_definition_embedding = embedding(industry_text)
    industries[industry] = {'text':industry_text, 'vector':industry_definition_embedding}

data3 = {"rows": industries}
industries_definition_embedding_filepath = "../raw_data/industry_definition_embedding.json"
with open(industries_definition_embedding_filepath, "w", encoding='utf-8') as f:
    json.dump(data3, f, indent=4, ensure_ascii=False)

