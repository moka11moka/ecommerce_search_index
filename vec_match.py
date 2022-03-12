import collections
import numpy as np
from tqdm import tqdm
import os
from zhconv import convert
from annoy import AnnoyIndex



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

def get_max_score(q_v, vectors):
    scores = [(-1, -1), (-1, -1), (-1, -1)]
    for i, vec in enumerate(vectors):
        score = np.dot(q_v, vec) / (np.linalg.norm(q_v) * np.linalg.norm(vec))
        if score > scores[0][1]:
            scores = [(i, score), scores[1], scores[2]]
        elif score > scores[1][1]:
            scores = [scores[0], (i, score), scores[2]]
        elif score > scores[2][1]:
            scores = [scores[0], scores[1], (i, score)]
    scores = [(i, s) for i, s in scores if i != -1]
    return scores


def ann_build(total_vecs, dim=128):
    annoyIndex = AnnoyIndex(dim, 'angular')
    for i in range(total_vecs.shape[0]):
        annoyIndex.add_item(i, total_vecs[i])
    annoyIndex.build(-1)
    return annoyIndex



if __name__ == '__main__':
    corpos_path = './datasets_tianchi/corpus.tsv'
    query_path = './datasets_tianchi/dev.query.txt'
    ind2doc = get_doc_index(corpos_path)
    ind2qry = get_doc_index(query_path)

    query_emb_path = './query_embedding'
    doc_emb_path = './doc_embedding'
    query_embed = read_vec(query_emb_path)
    doc_embed = read_vec(doc_emb_path)

    ind2docs = list(doc_embed.items())
    doc_vecs = np.array(list(doc_embed.values()))
    print(f"The shape of doc vector:{doc_vecs.shape}")
    annoyIndex = ann_build(doc_vecs)

    qry_doc = collections.defaultdict(list)
    for qry_index, qry_emb in tqdm(list(query_embed.items())[:4]):
        topk_result = annoyIndex.get_nns_by_vector(qry_emb, 20480, include_distances=True)[0]
        scores = get_max_score(qry_emb, doc_vecs[topk_result])
        for i, score in scores:
            index = topk_result[i]
            doc = ind2docs[index][0]
            qry_doc[qry_index].append((doc, score))
    
    for qr_in, docs in qry_doc.items():
        for doc_in, score in docs:
            print(f'{ind2qry[qr_in]} ==> {ind2doc[doc_in]} ==> {score}')
        print('==='*20)





