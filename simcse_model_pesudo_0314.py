import os
from tqdm import tqdm
import numpy as np
import random
from zhconv import convert
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import collections
from tensorflow.keras.utils import to_categorical

from transformers import  TFBertModel, BertConfig, BertTokenizer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# modelname = 'hfl/chinese-roberta-wwm-ext-large'
# tokenizer = BertTokenizer.from_pretrained(modelname)
# bert = TFBertModel.from_pretrained(modelname)

modelname = '/data/guoxiang/models/simbert'
config_path = os.path.join(modelname, 'config.json')
checkpoint_path = os.path.join(modelname, 'bert_model.ckpt.index')


tokenizer = BertTokenizer.from_pretrained(modelname)
config = BertConfig.from_json_file(config_path)
bert = TFBertModel.from_pretrained(modelname, from_pt=True)


max_len = 32
dimension = 128

class DataLoader(object):
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.index_query = './datasets_tianchi/train.query.txt'
        self.index_doc = './datasets_tianchi/corpus.tsv'
        self.index_dev = './datasets_tianchi/dev.query.txt'
        self.query_to_doc = './datasets_tianchi/train.query.99.tsv'
        self.doc_to_doc = './datasets_tianchi/doc2.match_10.txt'
        self.query_label = './datasets_tianchi/train_labeled.query.txt'
        self.steps = 0
        
    def data_preprocess(self, sentence:str):
        sentence = convert(sentence, 'zh-cn')
        return sentence

    def get_query_label(self, file_path:str):
        qry_label = {}
        with open(file_path) as f:
            for line in tqdm(f):
                line = line.strip().split('\t')
                index, label = line[0], line[-2]
                qry_label[index] = label
        return qry_label

    def read_negative_samples(self, file_path:str):
        doc2doc = {}
        if os.path.exists(file_path):
            with open(file_path) as f:
                for line in tqdm(f):
                    line = line.strip().split('\t')
                    index, match_index = line[0], line[1].split(',')
                    doc2doc[index] = match_index
        else:
            ValueError(f"No such file {file_path}")
        return doc2doc


    def read_index_data(self, file_path:str):
        if os.path.exists(file_path):
            index_sentence = {}
            with open(file_path) as f:
                for line in tqdm(f):
                    line = line.strip().split('\t')
                    index, sentence = line[0], "".join(line[1:])
                    index_sentence[index] = self.data_preprocess(sentence)
                    if len(index_sentence) == 100000:
                        break
            return index_sentence
        else:
            ValueError(f"Not file path {file_path}")
    def data_loader(self):
        print('reading query....')
        self.ind2query = self.read_index_data(self.index_query)
        print('reading doc....')
        self.ind2doc = self.read_index_data(self.index_doc)
        self.ind2dev = self.read_index_data(self.index_dev)
        print('reading query to doc....')
        self.qr2doc = self.read_index_data(self.query_to_doc)
        self.doc2qr = {d:q for q, d in self.qr2doc.items()}
        print('reading query to negative doc....')
        # self.doc2doc = self.read_negative_samples(self.doc_to_doc)
        print('reading query to label....')
        
        # self.qry2label = self.get_query_label(self.query_label)
        # self.steps = len(self.qr2doc)//self.batch_size + 1 if len(self.qr2doc) % self.batch_size else len(self.qr2doc)//self.batch_size
        leng = len(list(self.ind2doc.items())+list(self.ind2query.items()) + list(self.ind2dev.items()))
        self.steps = leng//self.batch_size + 1 if leng % self.batch_size else leng //self.batch_size
        print(f'the steps is {self.steps}')
    
    def data_generator(self):
        batch_tokens = []
        for i, (index, sent) in enumerate(list(self.ind2doc.items())+list(self.ind2query.items()) + list(self.ind2dev.items())):
            
            input_ids = tokenizer.encode(sent, add_special_tokens=True)
            batch_tokens.append(input_ids)
                # input_ids = tokenizer.encode(self.doc2qr[sent], add_special_tokens=True)
            batch_tokens.append(input_ids)

            if (len(batch_tokens) == 2 * self.batch_size) or (i == len(self.ind2doc) - 1):
                batch_token_ids = pad_sequences(batch_tokens, maxlen=max_len, padding='post')
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield batch_token_ids, batch_labels
                batch_tokens = []
    
    def forfit(self):
        while True:
            for data in self.data_generator():
                yield data

class Loss(Layer):
    def __init__(self, output_axis=None, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_axis = output_axis

    def call(self, inputs, mask=None):
        loss = self.compute_loss(inputs)
        self.add_loss(loss)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, list):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs):
        raise NotImplementedError

class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。"""
    def compute_loss(self, y_true, y_pred, alpha=0.5, beta=0.5):
        print("The y_true and y_pred shape in compute loss ", y_true.shape, y_pred.shape)
        loss1 = self.hard_loss(y_true, y_pred)
        return loss1

    def hard_loss(self, y_true, y_pred):
        """用于SimCSE训练的loss"""
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = K.equal(idxs_1, idxs_2)
        y_true = K.cast(y_true, K.floatx())
        # 计算相似度
        y_pred = K.l2_normalize(y_pred, axis=1)
        similarities = K.dot(y_pred, K.transpose(y_pred))
        similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
        similarities = similarities * 20
        loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
        return loss


def simcse_model():
    input_ids = Input(shape=(max_len, ), dtype=tf.int32, name='input_ids')
    bert = TFBertModel.from_pretrained(modelname, from_pt=True)
    last_layer = bert.bert(input_ids)
    logits = last_layer.last_hidden_state[:,0,:] # shape is ( max_len, hidden_size)
    output = Dense(units=dimension)(logits)
    print("The last hidden layer:", output.shape)
    model = Model(inputs=input_ids, outputs=output)
    model.summary()
    return model


if __name__ == '__main__':
    dl = DataLoader()
    dl.data_loader()
    data_gene = dl.forfit()
    
    total_loss = TotalLoss()
    learning_rate = 1e-5
    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-08)
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1","/gpu:2"])
    with mirrored_strategy.scope():
        model = simcse_model()
        model.compile(loss=total_loss.hard_loss, optimizer=optimizer)
        model.fit(data_gene, steps_per_epoch=dl.steps, verbose=1, epochs=1)
        model.save('./simbert_selfsup_alldata_model')
