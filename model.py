import torch
from transformers import AutoModel
import torch.nn.functional as F
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


class TextBackbone(torch.nn.Module):
    def __init__(self,
                 pretrained='/data/guoxiang/models/simbert',
                 output_dim=128):
        super(TextBackbone, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained).cuda()
        self.drop = torch.nn.Dropout(p=0.2)
        self.fc = torch.nn.Linear(768, output_dim)
        # self.fc1 = torch.nn.Linear(output_dim, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.extractor(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)

        first = out.hidden_states[1].transpose(1, 2)
        last = out.hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(
            first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(
            -1)  # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)),
                        dim=1)  # [batch, 2, 768]
        out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        # x = self.drop(out)
        x = self.fc(out)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def predict(self, x):
        x["input_ids"] = x["input_ids"].squeeze(1)
        x["attention_mask"] = x["attention_mask"].squeeze(1)
        x["token_type_ids"] = x["token_type_ids"].squeeze(1)
        out = self.extractor(**x, output_hidden_states=True)
        first = out.hidden_states[1].transpose(1, 2)
        last = out.hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(
            first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(
            -1)  # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)),
                        dim=1)  # [batch, 2, 768]
        out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        x = self.fc(out)
        x = F.normalize(x, p=2, dim=-1)
        return x

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
            for name, param in self.model.named_parameters():
                if param.requires_grad and emb_name in name:
                    if is_first_attack:
                        self.emb_backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = alpha * param.grad / norm
                        param.data.add_(r_at)
                        param.data = self.project(name, param.data, epsilon)
	
    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]