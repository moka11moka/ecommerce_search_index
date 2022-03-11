import numpy as np
from tqdm import tqdm
import os
from zhconv import convert

def read_vec(file_path:str):
    query_embed = {}
    with open(file_path) as f:
        for line in tqdm(f):
            index, embs = line.split('\t')
            embedding = [float(v) for v in embs.split(',')]
            query_embed[index] = embedding
    return query_embed

def data_preprocess(sentence:str):
    sentence = convert(sentence.lower(), 'zh-cn')
    return sentence

def get_doc_index(file_path:str):
    if os.path.exists(file_path):
        index2sent, sent2index = {}, {}
        with open(file_path) as f:
            for line in tqdm(f):
                line = line.strip().split('\t')
                index, sentence = line[0], "".join(line[1:])
                sent = data_preprocess(sentence)
                index2sent[index] = sent
    return index2sent

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == '__main__':
    corpos_path = './datasets_tianchi/corpus.tsv'
    query_path = './datasets_tianchi/test.query.txt'
    ind2doc = get_doc_index(corpos_path)
    ind2qry = get_doc_index(query_path)

    query_emb_path = './test_embedding'
    doc_emb_path = './doc_embedding'
    query_embed = read_vec(query_emb_path)
    doc_embed = read_vec(doc_emb_path)

    qry_doc = {}
    for qry_index, qry_emb in tqdm(list(query_embed.items())[:3]):
        score = -1
        for doc_index, doc_emb in doc_embed.items():
            val = cosine_similarity(qry_emb, doc_emb)
            if qry_index=='1':
                print(val)
            if val > score:
                score = val
                qry_doc[qry_index] = (doc_index, score)
    for qry_index, doc_index in qry_doc.items():
        doc_inx, score = doc_index
        print(f'{ind2qry[qry_index]} ===> {ind2doc[doc_inx]} ===> {score}')    
