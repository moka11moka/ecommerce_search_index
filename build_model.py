import os
from tqdm import tqdm
import numpy as np
import random
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
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

modelname = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(modelname)
bert = TFBertModel.from_pretrained(modelname)
max_len = 64

class DataLoader(object):
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.index_query = './datasets_tianchi/train.query.txt'
        self.index_doc = './datasets_tianchi/corpus.tsv'
        self.query_to_doc = './datasets_tianchi/qrels.train.tsv'
        self.tokenizer = BertTokenizer.from_pretrained(modelname)
        self.steps = 0

        # data preprocess (繁体转简体)
    def data_preprocess(self, sentence:str):
        sentence = convert(sentence.lower(), 'zh-cn')
        return sentence
            
    def read_index_data(self, file_path:str, sep_str='\t'):
        if os.path.exists(file_path):
            index_sentence = {}
            with open(file_path) as f:
                for line in tqdm(f):
                    line = line.strip().split('\t')
                    index, sentence = line[0], "".join(line[1:])
                    index_sentence[index] = self.data_preprocess(sentence)
            return index_sentence
        else:
            ValueError(f"Not file path {file_path}")
    def data_loader(self):
        print('reading query....')
        self.ind2query = self.read_index_data(self.index_query)
        print('reading doc....')
        self.ind2doc = self.read_index_data(self.index_doc)
        print('reading query to doc....')
        self.qr2doc = self.read_index_data(self.query_to_doc)
        self.steps = len(self.qr2doc)//self.batch_size + 1 if len(self.qr2doc) % self.batch_size else len(self.qr2doc)//self.batch_size
        print(f'the steps is {self.steps}')
    
    def get_labels(self, dim, step=3):
        rows = list(range(dim))
        cols = [i for i in range(0, step*dim ,2)]
        labels = np.zeros((dim, 2*dim))
        for i,j in zip(rows, cols):
            labels[(i, j)] = 1
        return labels

    def data_generator(self):
        query_ins = list(self.ind2query.keys())
        label_doc_inds = list(self.qr2doc.values())
        all_doc_ins = list(self.ind2doc.keys())
        print("read negative samples....")
        doc_ins = collections.Counter(all_doc_ins + label_doc_inds)
        negs_ins = []
        for index, num in tqdm(doc_ins.items()):
            if num < 2:
                negs_ins.append(index)
        batch_tokens = []
        for i, (qr_in, doc_in) in enumerate(self.qr2doc.items()):
            query, doc = self.ind2query[qr_in], self.ind2doc[doc_in]
            neg_doc = self.ind2doc[random.sample(negs_ins, 1)[0]]
            batch_tokens.append(query)
            batch_tokens.append(doc)
            batch_tokens.append(neg_doc)
            if (len(batch_tokens) == 3 * self.batch_size) or (i == len(self.qr2doc)-1):
                batch_token_ids = tokenizer(batch_tokens, add_special_tokens=True, padding=True, truncation=True, max_length=max_len, return_tensors='tf')
                # labels = to_categorical(list(range(self.batch_size*3)), self.batch_size)
                labels = self.get_labels(dim=len(batch_tokens)//3)
                yield [batch_token_ids['input_ids'],batch_token_ids['token_type_ids'], batch_token_ids['attention_mask']] , labels
                batch_tokens = []
    
    def forfit(self):
        while True:
            for data in self.data_generator():
                yield data

def get_labels(dim, step=3):
    rows = list(range(dim))
    cols = [i for i in range(0, step*dim ,2)]
    labels = np.zeros((dim, 2*dim))
    for i,j in zip(rows, cols):
        labels[(i, j)] = 1
    return labels

def simcse_loss(y_true, y_pred):
    y_true = get_labels(int(y_pred.shape[0])//3)
    y_pred = K.l2_normalize(y_pred, axis=1)
    cos_sim = K.dot(y_pred, K.transpose(y_pred))
    rows = tf.range(0, y_pred.shape[0], 3)
    col = tf.range(y_pred.shape[0])
    col = tf.where(tf.math.floormod(col, 3)!=0)
    cols = tf.transpose(col)[0]
    cos_sim = tf.gather(cos_sim, indices=rows)
    cos_sim = tf.gather(cos_sim, indices=cols, axis=1)
    loss = K.categorical_crossentropy(y_true, cos_sim, from_logits=True)
    return K.mean(loss)

def build_model():
    input_ids = Input(shape=(max_len,), name='input_ids', dtype='int32')
    token_type_ids = Input(shape=(max_len,), name='token_type_ids', dtype='int32')
    attention_mask = Input(shape=(max_len,), name='attention_mask', dtype='int32')
    bert_model = bert([input_ids, token_type_ids, attention_mask])
    cls_output = bert_model.last_hidden_state[:,0,:]
    logits = Dense(128)(cls_output)
    model = Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=logits, name='Bert_CL')
    model.summary()
    return model

if __name__ == '__main__':
    dl = DataLoader()
    dl.data_loader()
    data_gene = dl.forfit()
    checkpoints_path = 'checkpoints/checkpoint.ckpt'
    checkpoint_callback =  tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path,
                                                monitor='loss',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1,
                                                period=1)
    # mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])
    # with mirrored_strategy.scope():
    model = build_model()
    optimizer = Adam(learning_rate=1e-5, epsilon=1e-08)
    model.compile(loss=simcse_loss, optimizer=optimizer, run_eagerly=True)
    model.fit(data_gene, steps_per_epoch=dl.steps, epochs=20, callbacks=[checkpoint_callback])
    # model.save('models')



    