import string
import re
import itertools
import torch
import numpy as np
import random

# shamelessly adapted from https://rguigoures.github.io/word2vec_pytorch/
def load_brown_corpus():
    import nltk
    nltk.download('brown')
    from nltk.corpus import brown

    corpus = []
    for cat in brown.categories():
        for text_id in brown.fileids(cat):
            sentences = []
            for sent in brown.sents(text_id):
                text = ' '.join(sent)
                text = text.lower()
                for punct in string.punctuation:
                    text.replace(punct, ' ')
                text = re.sub('[^a-z.,0-9 ]+', '', text)

                tokens = [w for w in text.split() if w != '']
                if len(tokens) == 0:
                    continue
                if tokens[-1] == '.':
                    del tokens[-1]
                tokens.append('<eos>')

                sentences.append(tokens)
            corpus.append(sentences)

    # list of sentences (which are lists of words)
    corpus = list(itertools.chain.from_iterable(corpus))

    return corpus

# try to make as reproducible as possible
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
assert(torch.cuda.is_available())
