from transformers import BertForMaskedLM,BertTokenizerFast
import pandas as pd
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from transformers import LineByLineTextDataset
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--train_input', default='/home/hdp-portrait/fast-data/pre_train_corpus.csv', help="train input directory")
parser.add_argument('--test_input', default='', help="test input directory")
parser.add_argument('--add_toekns', default='/home/hdp-portrait/fast-data/add_tokens.txt', help="add_token directory")
parser.add_argument('--output_path', default='/home/hdp-portrait/data-vol-polefs-1/litiancheng/tianchi/checkpoints', help="checkpoint directory")
parser.add_argument('--init_checkpoint', default='/home/hdp-portrait/data-vol-polefs-2/model/hfl_roberta_base', help="init model checkpoint")
parser.add_argument('--vocab_path', default='/home/hdp-portrait/data-vol-polefs-1/litiancheng/tianchi/tokenizer', help="vocab directory")
parser.add_argument('--max_seq_length', default=100,help='max sequence length', type=int)
parser.add_argument('--train_batch_size', default=512,help='batch size', type=int)
parser.add_argument('--num_epochs', default=40,help='num of epochs', type=int)
parser.add_argument('--learning_rate', default=3e-5,help='learning rate', type=float)
parser.add_argument('--weight_decay', default=0.01,help='weight decay', type=float)
parser.add_argument('--warmup_proportion', default=0.1,help='proportion of warmup', type=float)

class Pretrain_Transformer(object):
    def __init__(self, maxlen=100, models_path=None,add_tokens=None, batch_size=512, num_epochs=40,lr=3e-5,weight_decay=0.01,warmup_proportion=0.1):
        self.maxlen = maxlen
        self.tokenizer, self.model = self.init_vocab_model(models_path,add_tokens)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.warmup_proportion = warmup_proportion

    # def input_fn(self, train_file):
    #     df = pd.read_csv(train_file, names=['text'], sep='\t', header=None)
    #     dataset = Dataset.from_pandas(df)
    #     return dataset
    def input_fn(self, train_file):
        dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=train_file,
            block_size=self.maxlen,
        )
        return dataset

    def init_vocab_model(self,models_path,add_tokens):
        tokenizer = BertTokenizerFast.from_pretrained(str(models_path))
        model = BertForMaskedLM.from_pretrained(str(models_path))
        new_tokens = pd.read_csv(add_tokens,names=['word'],header=None,sep='\t')['word'].tolist()
        print(f"vocab size before add tokens:{len(tokenizer)}")
        tokenizer.add_tokens(new_tokens)
        print(f"vocab size after add tokens:{len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    def tokenize(self, dataset):
        encodings = dataset.map(lambda e: self.tokenizer(e['text']), batched=True,batch_size=1000,num_proc=4, remove_columns=["text"])
        return encodings

    def train(self,train_input,test_input,output_path):
        train_encodings = self.input_fn(train_input)
        test_encodings = self.input_fn(test_input)
        #train_encodings = self.tokenize(train_dataset)
        steps = int(1002500 / self.batch_size * self.num_epochs)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        args = TrainingArguments(
            evaluation_strategy = "epoch",
            save_strategy="epoch",
            output_dir=output_path,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            warmup_steps=int(steps*self.warmup_proportion)
        )
        trainer = Trainer(
            self.model,  ## the instantiated 🤗 Transformers model to be trained
            args,  # # training arguments, defined above
            train_dataset=train_encodings,
            eval_dataset=test_encodings,
            data_collator=data_collator,
            tokenizer=self.tokenizer
            #     compute_metrics=compute_metrics
        )
        trainer.train()
        
    def save_model(self, path):
        self.model.save_pretrained(path)

    def save_vocabulary(self, path):
        self.tokenizer.save_pretrained(path)

if __name__ == '__main__':
    args = parser.parse_args()
    clf = Pretrain_Transformer(maxlen=args.max_seq_length,
                               models_path=args.init_checkpoint,
                               add_tokens=args.add_toekns,
                               batch_size=args.train_batch_size,
                               num_epochs=args.num_epochs,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay,
                               warmup_proportion=args.warmup_proportion)
    start = time.time()
    clf.save_vocabulary(args.vocab_path)
    clf.train(args.train_input,args.test_input,args.output_path)
    end = time.time()
    print(f'used {str((end - start) / 60)} mins')