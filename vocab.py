import string
import pickle


NUM = "<NUM>"
PUNCTUATION = "<PUNCT>"
UNK = "<UNK>"


class Vocab:

    def __init__(self, punct=None):
        self.words = None
        self.word2index = None
        if punct is None:
            self.punct = string.punctuation
        else:
            self.punct = punct

    def load_dictionary(self, path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        self.words = model.words
        self.punct = model.punct
        self.word2index = model.word2index
        self.unk_index = model.unk_index
        self.padding_index = model.padding_index
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    def process_word(self, word, lowercase=True):
        if lowercase:
            word = word.lower()
        if word[0].isdigit():
            word = NUM
        elif word in self.punct:
            word = PUNCTUATION
        else:
            flag = True
            for c in word:
                if c in self.punct:
                    continue
                else:
                    flag = False
            if flag is True:
                word = PUNCTUATION
        return word
    
    def word_to_index(self, word, lowercase=True):
        word = self.process_word(word, lowercase)
        return self.word2index.get(word)
    
    def build(self, train):
        if self.words is None:
            self.words = set()
        for sent in train:
            for word in sent:
                word = self.process_word(word, lowercase=True)
                if word in self.words:
                    continue
                else:
                    self.words.add(word)
        self.word2index = {word: i for i, word in enumerate(self.words)}
        self.unk_index = len(self.words)
        self.padding_index = len(self.words) + 1


if __name__ == "__main__":
    with open("X.pkl", "rb") as f:
        train = pickle.load(f)
    print(len(train))
    with open("punctuation.pkl", "rb") as f:
        punct = pickle.load(f)
    vocab = Vocab(punct=punct)
    # vocab.build(train)
    vocab.load_dictionary("vocab.pkl")
    print(len(vocab.words))