import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial.distance import cosine
import json

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["rows"]

def embedding(text):
    if isinstance(text, str):
        text = [text]
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda:0')
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    text_embeddings = model_output[0][:, 0]
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
    return text_embeddings[0].cpu().tolist()

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

industry_definition_embedding_filepath = "../raw_data/industry_definition_embedding.json"
industry_definition_embedding_data = load_json_data(industry_definition_embedding_filepath)
industry_definition_embedding_data_index = {index: industry for index, industry in enumerate(industry_definition_embedding_data)}

def find_similar_vectors(data, query_vector, top_k=5):    
    similarities = []
    indices = []
    
    for index, row in enumerate(data):
        vector = np.array(data[row]['vector'])
        similarity = cosine_similarity(query_vector, vector)
        similarities.append(similarity)
        indices.append(index)
    
    # 获取相似度最高的 top_k 个索引
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # 获取 top_k 个最相似的结果
    top_k_results = [
        {
            "industry": industry_definition_embedding_data_index[i], 
            "desc": data[industry_definition_embedding_data_index[i]]['text'], 
            "distance": similarities[i]
        }  for i in top_k_indices ]
    
    return top_k_results


df = pd.read_excel('../processed_data/all_bond_company_data.xlsx', sheet_name='all')

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model = model.to(torch.device('cuda:0'))
model.eval()

output = {
    "results": []
}
company_short_list = []

# short - 经营范围
for i in tqdm(range(len(df))):
    name, text = df.iloc[i, 4], df.iloc[i, 2]
    if name in company_short_list:
        continue
    company_short_list.append(name)
    text_embedding = embedding(text)
    similarity = find_similar_vectors(industry_definition_embedding_data, text_embedding)
    output["results"].append({
        "company": name,
        "desc": text,
        "similarity": similarity
    })

import json
with open("../processed_data/company2industry.json", "w", encoding='utf-8') as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(len(output["results"]))