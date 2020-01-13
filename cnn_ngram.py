import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
from torch.nn.utils import clip_grad_norm_
import parser
import torch
import os


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path):
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.dictionary.add_word(word)  
        
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids


class BrownDataset(Dataset):
    def __init__(self, corpus_file, seq_length, device):
        self.corpus_file = corpus_file
        self.seq_length = seq_length
        self.corpus = Corpus()
        self.ids = self.corpus.get_data(corpus_file)
        self.device = device
        self.vocab_size = len(self.corpus.dictionary)
        
    def id_to_word(self, _id: int):
        return self.corpus.dictionary.idx2word[_id]

    def __len__(self):
        # -1 for the target
        return len(self.ids) - self.seq_length - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        _input = self.ids[idx:idx+self.seq_length]
        _target = self.ids[idx+self.seq_length]

        return _input.to(self.device), _target.to(self.device)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv1dBlockBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel,
                      kernel_size=kernel_size, stride=stride),
            nn.Dropout(p),
            nn.PReLU(),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class CNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_length):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.conv1 = Conv1dBlockBN(embed_size, 32, kernel_size=5, stride=1)
        self.conv2 = Conv1dBlockBN(32, 8, kernel_size=3, stride=1)
        self.linear = nn.Linear(8*(seq_length-4-2), vocab_size)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_size = args.embed_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seq_length = args.seq_length
    learning_rate = args.learning_rate

    file = '/home/aab11165ig/language_model/data/browncorpus.txt'
    dataset = BrownDataset(file, seq_length, device)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True)

    vocab_size = dataset.vocab_size
    model = CNNLM(vocab_size, embed_size, seq_length).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for batch, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            model.zero_grad()
            loss.backward()
            # clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if batch % 100 == 0:
                print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                    .format(epoch+1, num_epochs, batch, len(dataloader), loss.item(), np.exp(loss.item())))


    # Save the model checkpoints
    torch.save(model.state_dict(), '/home/aab11165ig/language_model/data/cnn_model.ckpt')
