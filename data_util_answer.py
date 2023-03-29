from torch.utils.data import DataLoader, Dataset
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch.nn.functional as F
import torch
import random




class MyDataset(Dataset):
    def __init__(self, filename, max_length=64, train=True, max_example_num=None, random_state=0):
        self.max_length = max_length
        self.text, self.label = self.load(filename, train=train)
        text_ids = batch_to_ids(self.text)
        self.text = self.pad(text_ids).tolist()
        if max_example_num:
            random.seed(0)
            sampled_index = random.sample(range(len(self.text)), min(max_example_num, len(self.text)))
            self.text = [self.text[i] for i in sampled_index]
            self.label = [self.label[i] for i in sampled_index]
    
    def load(self, file, train):
        text, label = [], []
        f = open(file, "r+", encoding="utf-8")
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            content = line.strip().split("\t")
            if train:
                _label = content[1].strip()
                _text = content[0].strip().split(" ")
            else:
                _label = content[0].strip()
                _text = content[1].strip().split(" ")
            label.append(int(_label))
            text.append(_text)
        return text, label
    
    def pad(self, text_ids):
        return F.pad(text_ids, pad=(0,0,0,self.max_length-text_ids.size(1),0,0), mode='constant', value=0)


    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item], self.label[item]


def collate_fn(data, device):
    text, label = [], []
    for _text, _label in data:
        label.append(_label)
        text.append(_text)
    return torch.tensor(text).to(device), torch.tensor(label).to(device)

if __name__ == "__main__":
    dataset = MyDataset("./data/dev.tsv")
    # print(dataset[0])