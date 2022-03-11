from tqdm import tqdm
import numpy as np
import random
from typing import List, Dict
from zhconv import convert
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import collections
from tensorflow.keras.utils import to_categorical

from transformers import  TFBertModel, BertConfig, BertTokenizer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
cpkt_path = 'checkpoints/checkpoint.ckpt'

max_len = 64
modelname = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(modelname)
bert = TFBertModel.from_pretrained(modelname)

def build_model():
    input_ids = Input(shape=(max_len,), name='input_ids', dtype='int32')
    token_type_ids = Input(shape=(max_len,), name='token_type_ids', dtype='int32')
    attention_mask = Input(shape=(max_len,), name='attention_mask', dtype='int32')
    bert_model = bert([input_ids, token_type_ids, attention_mask])
    cls_output = bert_model.last_hidden_state[:,0,:]
    logits = Dense(128)(cls_output)
    model = Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=logits, name='Bert_CL')
    model.load_weights(cpkt_path)
    model.summary()
    return model

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
                sent2index[sent] = index
    return index2sent, sent2index

def get_vec(model, sents:List[str], batch_size=2000):
    res_vecs = []
    for i in tqdm(range(0, len(sents), batch_size)):
        batch_sents = sents[i: min(i+batch_size, len(sents))]
        batch_token_ids = tokenizer(batch_sents, add_special_tokens=True, padding=True, truncation=True, max_length=max_len, return_tensors='tf')
        inputs = [batch_token_ids['input_ids'],batch_token_ids['token_type_ids'], batch_token_ids['attention_mask']]
        vecs = model(inputs).numpy()
        res_vecs.extend(vecs)
    return res_vecs

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


if __name__ == '__main__':
    file_path = './datasets_tianchi/test.query.txt'
    model = build_model()
    index2sent, sent2index = get_doc_index(file_path)
    print(list(index2sent.values())[:5])
    sents = list(index2sent.values())
    vecs = get_vec(model, sents)
    queries = []
    for i, vec in enumerate(tqdm(vecs)):
        sent = sents[i]
        index = sent2index[sent]
        vec = ",".join([str(v) for v in vec])
        line = index + '\t' + vec + '\n'
        queries.append(line)
    queries.sort()
    with open('test_embedding', 'a+') as f:
        for line in queries:
            f.write(line)
