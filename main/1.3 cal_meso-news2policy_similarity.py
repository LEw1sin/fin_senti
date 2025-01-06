import json
from tqdm import tqdm
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial.distance import cosine

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["rows"]

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def find_similar_vectors(data, query_vector, top_k=5):    
    # 初始化相似度列表和索引列表
    similarities = []
    indices = []
    
    for index, row in enumerate(data):
        vector = np.array(row['vector'])
        similarity = cosine_similarity(query_vector, vector)
        similarities.append(similarity)
        indices.append(index)
    
    # 获取相似度最高的 top_k 个索引
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # 获取 top_k 个最相似的结果
    top_k_results = [
        {
            "policy_name": data[i]['policy_name'], 
            "definition": data[i]['definition'], 
            "distance": similarities[i]
        }  for i in top_k_indices ]
    
    return top_k_results

if __name__ == "__main__":
    filepath_list = ["../raw_data/merged_data_except_2020.json", "../raw_data/data_2020.json"]
    for filepath in filepath_list:
        data = json.load(open(filepath, "r", encoding="utf-8"))
        # local cimpute
        json_path = '../raw_data/policy_definition_embedding.json' 
        policy_data = load_json_data(json_path)
        # load embeddings
        embeddings = np.load("../raw_data/embeddings_merged_data.npy")  # (766778, 1024)
        print(embeddings.shape)

        for i in tqdm(range(len(data))):
            # print(data[i])
            text_embedding = embeddings[i].tolist()
            # policies = search_policy(text_embedding)
            policies = find_similar_vectors(policy_data, text_embedding)
            data[i]["policies"] = policies

        with open("../raw_data/merged_data_except_2020_similarity.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


