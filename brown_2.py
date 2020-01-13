import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
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
CONTEXT_WIDTH = 16
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
tdata, labels = list(zip(*indexed_corpus))

class BrownDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.LongTensor(data).to(device)
        self.labels = torch.LongTensor(labels).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = BrownDataset(tdata, labels)

BATCH_SIZE = 256
train_dataset, val_dataset = data.random_split(dataset, [TRAIN_SIZE, SAMPLE_SIZE - TRAIN_SIZE])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class CNN_LM(nn.Module):
    def __init__(self, char_vocab_size, embed_dim, context_width, chan_size, hid_size, vocab_size):
        super().__init__()
        self.embedding =  nn.Embedding(char_vocab_size, embed_dim)
        convs = []
        for i in range(context_width - 1):
            if i == 0:
                convs.append(nn.Conv1d(embed_dim, chan_size, 2))
            else:
                convs.append(nn.Conv1d(chan_size, chan_size, 2))
        self.convs = nn.Sequential(*convs)
        self.fc1 = nn.Linear(chan_size, hid_size)
        self.fc2 = nn.Linear(hid_size, vocab_size)

    def forward(self, context):
        x = self.embedding(context).permute(0, 2, 1) # (batch_size, embed_dim, context_width)
        x = self.convs(x) # (batch_size, chan_size, 1)
        x = x.squeeze(2) # (batch_size, chan_size)
        x = F.relu(self.fc1(x)) # (batch_size, hid_size)
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

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

NUM_EPOCHS=25

for epoch in range(NUM_EPOCHS):
    print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
    print('-' * 10)

    # train
    model.train()

    running_loss = 0.0
    for inputs, labels in tqdm(train_dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    scheduler.step()
    epoch_loss = running_loss / (len(train_dataloader) * BATCH_SIZE)
    print('train loss: {}'.format(epoch_loss))

    # eval
    model.eval()

    running_loss = 0.0
    for inputs, labels in tqdm(val_dataloader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / (len(val_dataloader) * BATCH_SIZE)
    print('val loss: {}'.format(epoch_loss))

#import IPython
#IPython.embed()
