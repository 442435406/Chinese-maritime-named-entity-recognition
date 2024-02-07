import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def get_bert_embeddings(sentence, model, tokenizer):
    tokens = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(tokens)
    last_hidden_states = outputs.last_hidden_state
    pooled = last_hidden_states.mean(dim=1)
    return pooled

def cosine_similarity_between_sentences(sentence1, sentence2, model, tokenizer):
    embeddings1 = get_bert_embeddings(sentence1, model, tokenizer)
    embeddings2 = get_bert_embeddings(sentence2, model, tokenizer)
    similarity = cosine_similarity(embeddings1, embeddings2)
    return similarity[0][0]

# 加载BERT模型和tokenizer
model_name = 'bert-base-chinese'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 读取sentence1
sentence1 = "船舰舱海渔港"

# 从txt文件中读取sentence2
file_path = 'trans_clue_pretrain_0000000.txt'  # 替换为你的txt文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    sentences_from_file = [line.strip() for line in file]

# 计算余弦相似度并保存结果到DataFrame
results = {'Sentence2': sentences_from_file, 'Similarity': []}
for sentence2 in sentences_from_file:
    if (len(sentence2)<512):
        similarity_score = cosine_similarity_between_sentences(sentence1, sentence2, model, tokenizer)
        results['Similarity'].append(similarity_score)
        print(similarity_score)
    else:
        results['Similarity'].append("0")
# 创建DataFrame并输出到Excel文件
df = pd.DataFrame(results)
output_excel_path = 'output_results.xlsx'  # 替换为你想要保存的Excel文件路径
df.to_excel(output_excel_path, index=False)

print(f"Results saved to {output_excel_path}")
