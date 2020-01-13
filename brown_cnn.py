import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import string
import re
from collections import defaultdict
torch.manual_seed(0)
#np.random.seed(0)
random.seed(0)
assert(torch.cuda.is_available())
device = torch.device("cuda")
cpu = torch.device("cpu")
START_CHAR = chr(0)
CONTEXT_WIDTH = 10
VOCAB_SIZE=30000
browncorpus_path = "./browncorpus.txt"
def normalize_word(word):
for punct in string.punctuation:
word = word.replace(punct, '')
word = re.sub('[^a-z.,0-9 ]+', '', word)
return word
def make_context_word_pairs(sentence):
pairs = []
space_indices = [ -1 ] + [ pos for pos, char in enumerate(sentence) if char == " " ] + [ len(sentence) ]
for i in range(len(space_indices)):
word = ""
if i != len(space_indices) - 1:
word = normalize_word(sentence[space_indices[i] + 1: space_indices[i + 1]])
else:
word = "<eos>"
if word == "":
continue
context = ""
for i in range(max(0, space_indices[i] - CONTEXT_WIDTH), space_indices[i]):
context += sentence[i]
context = START_CHAR * (CONTEXT_WIDTH - len(context)) + context
assert(len(context) == CONTEXT_WIDTH)
pairs.append((context, word))
return pairs
def load_corpus(path):
with open(path, "r") as f:
pairs = []
for line in f:
sentence = line.strip().lower()
sentence = sentence.replace("  ", " ")
if sentence != "":
pairs += make_context_word_pairs(sentence)
return pairs
print("processing corpus...")
corpus = load_corpus(browncorpus_path)
words = list(map(lambda x: x[1], corpus))
word_counts = defaultdict(lambda: 0)
for word in words:
word_counts[word] += 1
words_sorted_by_freq = sorted([(v,k) for (k,v) in word_counts.items()])
words_sorted_by_freq.reverse()
words_sorted_by_freq = list(map(lambda kv: kv[1], words_sorted_by_freq))
i2w = words_sorted_by_freq[0:(VOCAB_SIZE-3)]
i2w.append('<start>')
i2w.append('<eos>')
i2w.append('<unk>')
START = VOCAB_SIZE-3
EOS = VOCAB_SIZE-2
UNK = VOCAB_SIZE-1
w2i = defaultdict(lambda: UNK, { k: v for (v, k) in enumerate(i2w)})
indexed_corpus = []
for context, word in corpus:
indexed_corpus.append((list(map(lambda ch: ord(ch), context)), w2i[word]))
print("done")
#print("shuffling...")
SAMPLE_SIZE = len(corpus)
TRAIN_SIZE = int(SAMPLE_SIZE * 0.8)
#random.shuffle(indexed_corpus)
#train_data, train_label = list(zip(*indexed_corpus[0:TRAIN_SIZE]))
#val_data, val_label = list(zip(*indexed_corpus[TRAIN_SIZE:SAMPLE_SIZE]))
#print("done")
data, labels = list(zip(*indexed_corpus))
class BrownDataset(Dataset):
def __init__(self, data, labels):
self.data = torch.LongTensor(data).to(device)
self.labels = torch.LongTensor(labels).to(device)
def __len__(self):
return len(self.labels)
def __getitem__(self, idx):
return self.data[idx], self.labels[idx]
dataset = BrownDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
class CNN_LM(nn.Module):
def __init__(self, char_vocab_size, embed_dim, context_width, chan_size, hid_size, vocab_size):
super().__init__()
self.embedding = nn.Embedding(char_vocab_size, embed_dim)
self.convs = []
for i in range(context_width - 1):
if i == 0:
self.convs.append(nn.Conv1d(embed_dim, chan_size, 2))
else:
self.convs.append(nn.Conv1d(chan_size, chan_size, 2))
self.fc1 = nn.Linear(chan_size, hid_size)
self.fc2 = nn.Linear(hid_size, vocab_size)
# TODO: check weight init
def forward(self, context, offsets):
x = self.embedding(context) # (batch_size, embed_dim, context_width)
for conv in self.conv:
x = conv(x) # (batch_size, chan_size, 1)
x = x.squeeze(2) # (batch_size, chan_size)
x = F.ReLU(self.fc1(x)) # (batch_size, hid_size)
x = self.fc2(x) # (batch_size, vocab_size)
return x
CHAR_VOCAB_SIZE=128
EMBED_DIM=100
CHAN_SIZE=50
HID_SIZE=50
model = CNN_LM(char_vocab_size=CHAR_VOCAB_SIZE, embed_dim=EMBED_DIM,
context_width=CONTEXT_WIDTH, chan_size=CHAN_SIZE, hid_size=HID_SIZE,
vocab_size=VOCAB_SIZE)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import IPython
IPython.embed()
