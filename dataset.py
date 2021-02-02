import const

from vocab import Vocab

import numpy as np
import pickle


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def padding(X, maxlen, value):
    for i, seq in enumerate(X):
        n = len(seq)
        if n == maxlen:
            continue
        elif n > maxlen:
            seq = seq[:n]
        else:
            for i in range(maxlen - n):
                seq.append(value)
        X[i] = np.array(seq)
    return np.array(X).reshape(len(X), maxlen)

def create_word_data(train):
    vocab = Vocab()
    vocab.load_dictionary("vocab.pkl")
    X = []
    y = []
    for sent in train:
        sent_int = []
        tag_int = []
        for words in sent:
            try:
                sent_int.append(vocab.word_to_index(words[0], lowercase=True))
            except KeyError:
                sent_int.append(vocab.unk_index)
            tag_int.append(const.WORD_TAG[words[-1]])
        X.append(sent_int)
        y.append(tag_int)
    X = padding(X, maxlen=const.MAX_LEN, value=vocab.padding_index)
    y = padding(y, maxlen=const.MAX_LEN, value=const.WORD_TAG["<PAD_WORD>"])
    y = to_categorical(y, 3)
    return X, y

if __name__ == "__main__":
    with open("test.pkl", "rb") as f:
        test = pickle.load(f)
    X_test, y_test = create_word_data(test)
    print(y_test[0])
