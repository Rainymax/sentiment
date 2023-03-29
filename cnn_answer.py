import numpy as np
import torch
from torch import nn
from allennlp.modules.elmo import Elmo


class TextCNN(nn.Module):
    def __init__(self, options_file, weight_file, vector_size, filter_size=[2,3,4,5], channels=64, max_length=1024):
        super(TextCNN, self).__init__()
        ####################
        self.embedding = Elmo(options_file, weight_file, 1, dropout=0)

        self.filter_size = filter_size
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(self.filter_size[i], vector_size), padding=0, stride=(1, 1)) for i in range(len(self.filter_size))])
        self.poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=max_length - self.filter_size[i] + 1, stride=1, padding=0) for i in range(len(self.filter_size))])
        self.transform = nn.Linear(len(self.filter_size) * channels, 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        #####################

    def forward(self, inputs):
        outputs = []
        inputs = self.embedding(inputs)["elmo_representations"][0]
        #############
        for i in range(len(self.filter_size)):
            conv_output = self.convs[i](inputs.unsqueeze(1))
            pooling_output = self.poolings[i](self.activation(conv_output.squeeze(-1)))
            outputs.append(pooling_output.squeeze(-1))
        return self.transform(self.dropout(torch.cat(outputs, dim=-1)))
        ##############