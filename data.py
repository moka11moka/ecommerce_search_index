import os
import random
from turtle import pos
import torch.utils.data as data
from transformers import AutoTokenizer
from zhconv import convert
import re
import jieba

class Unsupervised(data.Dataset):
    def __init__(self, root="/home/guoxiang/tf_tutoria/ecommerce_search/datasets_tianchi/") -> None:
        super(Unsupervised, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, "corpus.tsv")
        self.all_data = []
        self._create_train_data()
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/data/guoxiang/models/simbert')

    def _create_train_data(self):
        with open(self.all, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.all_data.append((line[1], line[1]))
                if len(self.all_data) >= 100000:
                    break

    def __getitem__(self, index):
        sample = self._post_process(self.all_data[index])
        return sample

    def _post_process(self, text):
        anchor = self.tokenizer([text, text],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=48,
                                padding='max_length',
                                return_tensors='pt').to("cuda:0")
        return anchor

    def __len__(self):
        return len(self.all_data)


class Supervised(data.Dataset):
    def __init__(self, root="/home/guoxiang/tf_tutoria/ecommerce_search/datasets_tianchi") -> None:
        super(Supervised, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, "corpus.tsv")
        self.train = os.path.join(self.root, "train.query.txt")
        self.corr = os.path.join(self.root, "qrels.train.tsv")
        self.split_char = "#@#@"
        self.all_data = {}
        self.train_data = {}
        self.neg_data = {}
        self._create_train_data()
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/data/guoxiang/models/simbert')

    def _create_train_data(self):
        with open(self.train, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.train_data[line[0]] = line[1] + self.split_char
        with open(self.all, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.all_data[line[0]] = line[1]
        with open(self.corr, 'r') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                k = line[0]
                v = line[1]
                self.train_data[k] += self.all_data[v]
        with open('/home/guoxiang/tf_tutoria/ecommerce_search/datasets_tianchi/doc2.match_10.txt') as f:
            for line in f:
                line = line.strip().split('\t')
                k = line[0]
                neg = line[1].split(',')[0]
                self.neg_data[self.all_data[k]] = self.all_data[neg]

    def __getitem__(self, index):
        index = str(index + 1)
        anchor_text, pos_text = self.train_data[index].split(self.split_char)
        tmp = random.randint(1, 1001492)
        neg_text = self.all_data[str(tmp)]
        # neg_text = self.neg_data[pos_text]

        sample = self._post_process(anchor_text, pos_text, neg_text)
        return sample

    def _post_process(self, anchor_text, pos_text, neg_text):
        sample = self.tokenizer([anchor_text, pos_text, neg_text],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=48,
                                padding='max_length',
                                return_tensors='pt').to("cuda:0")

        return sample

    def __len__(self):
        return len(self.train_data)


class TESTDATA(data.Dataset):
    def __init__(self, root="/home/guoxiang/tf_tutoria/ecommerce_search/datasets_tianchi/", certain="corpus.tsv") -> None:
        super(TESTDATA, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, certain)
        self.all_data = {}
        self._create_eval_data()
        self.start = 0
        self.length = 48
        if certain != "corpus.tsv":
            self.start = 200000
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/data/guoxiang/models/simbert')
        self.dtext = {"充气车库":"定制户外大型充气足球泡泡屋气模网红双层闭气帐篷民宿野营星空屋",
                    "兰芳园":"某大牌奶茶港式鸳鸯丝袜牛乳茶咖啡红茶280ml即饮杯装网红款整箱",
                    "农业电磁阀":"自动滴灌微喷农业草坪大棚工程浇水地埋式控制流量调节灌溉电磁阀",
                    "电脑袋":"2020新款笔记本内胆包适用于苹果Macbook13.3寸air联想小新戴尔华硕华为matebook14收纳包支架小米pro15mac套",
                    "新风机":"MATE非小米家新风机系统有品米皮A1免开洞换气通风除甲醛除雾霾",
                    "防夹手门挡":"免打孔门挡门楔子门塞挡门防撞顶门阻门器门吸卡门防风固定门阻",
                    "绿野仙踪英文版":"绿野仙踪The Wizard of Oz （全英文原版，世界经典英文名著文库，精装珍藏本）【果麦经典】",
                    "鞋子女":"乔丹板鞋2021冬季新款小白鞋时尚休闲鞋情侣撞色运动鞋男皮面女鞋",
                    "甲黄酸阿怕替尼片":"甲磺酸阿帕替尼片"}
                   
        self.stopwords = set()
        with open('/home/guoxiang/tf_tutoria/ecommerce_search/hit_stopwords.txt') as f:
            for line in f:
                self.stopwords.add(line.strip())

    def _create_eval_data(self):
        with open(self.all, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.all_data[line[0]] = line[1]

    def data_preprocess(self, sent:str):
        pattern = re.compile(r'&alpha;|&Iota;|&bull;|&deg;|&ldquo;|&ndash;|&rdquo;|~|&mdash;|&quot;|&amp;|&times;|&middot;|&rsaquo;|&upsilon;|&omicron;|&reg;|！|。|&pi;|&nbsp;')
        sent = re.sub(pattern, "", sent)
        return convert(sent.lower(), 'zh-cn')


    def __getitem__(self, index):
        id_, text = str(index + self.start +
                        1), self.all_data[str(index + self.start + 1)]
        text_pro = convert(text, 'zh-cn').replace(' ','').lower()
        if text_pro in self.dtext:
            text = self.dtext[text_pro]
        data = self.tokenizer(self.data_preprocess(text),
                              truncation=True,
                              add_special_tokens=True,
                              max_length=self.length,
                              padding='max_length',
                              return_tensors='pt').to("cuda:0")

        return id_, data

    def __len__(self):
        return len(self.all_data)