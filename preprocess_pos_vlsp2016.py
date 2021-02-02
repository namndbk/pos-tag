from os.path import dirname, join
import pickle


def pos_tag_normalize(tag):
    tags_map = {
        "Ab": "A",
        "B": "FW",
        "Cc": "C",
        "Fw": "FW",
        "Nb": "FW",
        "Ne": "Nc",
        "Ni": "Np",
        "NNP": "Np",
        "Ns": "Nc",
        "S": "Z",
        "Vb": "V",
        "Y": "Np"
    }
    if tag in tags_map:
        return tags_map[tag]
    else:
        return tag


def preprocess(sentences):
    def process_token(t):
        output = t[:2]
        output[0] = t[0].replace("_", " ")
        output[1] = pos_tag_normalize(t[1])
        return output

    def process_sentence(s):
        return [process_token(t) for t in s]

    return [process_sentence(s) for s in sentences]

import re
from itertools import chain
from os import makedirs, listdir
from os.path import dirname, join

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split


def load_dict(path):
    uni_grams, bi_grams, tri_grams = {}, {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tokens = line.strip().split()
                if len(tokens) == 1:
                    uni_grams[line.strip()] = 1
                elif len(tokens) == 2:
                    bi_grams[line.strip()] = 1
                elif len(tokens) == 3:
                    tri_grams[line.strip()] = 1
                else:
                    continue
    return uni_grams, bi_grams, tri_grams


def is_name(word):
    tokens = word.split(" ")
    for token in tokens:
        if not token.istitle():
            return False
    return True


def is_mix_case(word):
    return len(word) > 2 and word[0].islower() and word[1].isupper()


def get_word_shape(word):
    word_shape = []
    for character in word:
        if character.isupper():
            word_shape.append('U')
        elif character.islower():
            word_shape.append('L')
        elif character.isdigit():
            word_shape.append('D')
        else:
            word_shape.append(character)
    return ''.join(word_shape)


def is_date(word):
    return re.search(r"^([0-2]?[0-9]|30|31)[/-](0?[1-9]|10|11|12)([/-]\d{4})?$", word) is not None


def is_unigram(word, uni_grams):
    if uni_grams.get(word):
        return True
    else:
        return False


def is_bigram(word, bi_grams):
    if bi_grams.get(word):
        return True
    else:
        return False


def is_trigram(word, tri_grams):
    if tri_grams.get(word):
        return True
    else:
        return False


def is_cap_with_period(word):
    return word[0].isupper() and word[-1] == "."


def contains_hyphen(word):
    return "-" in word


def ends_with_digit(word):
    return word[-1].isdigit()


def is_range(word):
    if re.match(r"^\d+-\d+$", word) is not None:
        nums = re.split(r'-', word)
        first_num = int(nums[0])
        second_num = int(nums[1])
        return first_num < second_num and second_num - first_num < 1000

    return False


def is_rate(word):
    if re.match(r"^\d+/\d+$", word) is not None:
        nums = re.split(r'/', word)
        first_num = int(nums[0])
        second_num = int(nums[1])
        return first_num < second_num
    return False


def is_month_year(word):
    return re.match(r"^(0?[1-9]|11|12)[/-]\d{4}$", word) is not None


def is_code(word):
    return word[0].isdigit() and word[-1].isupper()


def digit_and_comma(word):
    return re.search(r"^\d+,\d+$", word) is not None


def digit_and_period(word):
    return re.search(r"^\d+\.\d+$", word) is not None


def word2features(sentence, i, uni_grams, bi_grams, tri_grams):
    word = sentence[i][0]

    features = {
        'w(0)': word,
        # 'w(0)[:1]': word[:1],
        # 'w(0)[:2]': word[:2],
        # 'w(0)[:3]': word[:3],
        # 'w(0)[:4]': word[:4],
        # 'w(0)[-1:]': word[-1:],
        # 'w(0)[-2:]': word[-2:],
        # 'w(0)[-3:]': word[-3:],
        # 'w(0)[-4:]': word[-4:],

        'word.islower': word.islower(),
        # 'word_in_dict': is_unigram(word, uni_grams),
        'word.lower': word.lower(),
        'isTitle': word.istitle(),
        'isNumber': word.isdigit(),
        'isUpper': word.isupper(),
        'isCapWithPeriod': is_cap_with_period(word),
        'endsWithDigit': ends_with_digit(word),
        'containsHyphen': contains_hyphen(word),
        'isDate': is_date(word) or is_month_year(word),
        'isCode': is_code(word),
        'isName': is_name(word),
        'isMixCase': is_mix_case(word),
        'd&comma': digit_and_comma(word),
        'd&period': digit_and_period(word),
        'wordShape': get_word_shape(word),

        'isRange': is_range(word),
        'isRate': is_rate(word),
    }

    if i > 0:
        previous_word = sentence[i - 1][0]

        features.update({
            'w(-1)': previous_word,
            'w(-1).lower': previous_word.lower(),
            'isTitle(-1)': previous_word.istitle(),
            'isNumber(-1)': previous_word.isdigit(),
            'isCapWithPeriod(-1)': is_cap_with_period(previous_word),
            'isName(-1)': is_name(previous_word),
            'wordShape(-1)': get_word_shape(previous_word),
            'w(-1)+w(0)': previous_word + ' ' + word,
        })

    else:
        features['BOS'] = True

    if i > 1:
        previous_word = sentence[i - 1][0]
        previous_2_word = sentence[i - 2][0]

        features.update({
            'w(-2)': previous_2_word,
            'w(-2)+w(-1)': previous_2_word + ' ' + previous_word,
            'isTitle(-2)': previous_2_word.istitle(),
            'isNumber(-2)': previous_2_word.isdigit(),
            # 'w(-2)+w(-1)+w(0)_in_dict': is_trigram(previous_2_word + ' ' + previous_word + ' ' + word, tri_grams)
        })

    if i < len(sentence) - 1:
        next_word = sentence[i + 1][0]

        features.update({
            'w(+1)': next_word,
            'w(+1).lower': next_word.lower(),
            'isTitle(+1)': next_word.istitle(),
            'isNumber(+1)': next_word.isdigit(),
            'isCapWithPeriod(+1)': is_cap_with_period(next_word),
            'isName(+1)': is_name(next_word),
            'wordShape(+1)': get_word_shape(next_word),
            'w(0)+w(+1)': word + ' ' + next_word,
            # 'w(0)+w(+1)_in_dict': is_bigram(word + ' ' + next_word, bi_grams)
        })
    else:
        features['EOS'] = True

    if i < len(sentence) - 2:
        next_word = sentence[i + 1][0]
        next_2_word = sentence[i + 2][0]

        features.update({
            'w(+2)': next_2_word,
            'w(+1)+w(+2)': next_word + ' ' + next_2_word,
            'isTitle(+2)': next_2_word.istitle(),
            'isNumber(+2)': next_2_word.isdigit(),
            # 'w(0)+w(+1)+w(+2)_in_dict': is_trigram(word + ' ' + next_word + ' ' + next_2_word, tri_grams)
        })

    return features


uni_grams, bi_grams, tri_grams = load_dict(
    "data/dictionary/pyvi_dictionary.txt")


def sent2features(sentence):
    return [word2features(sentence, i, uni_grams, bi_grams, tri_grams) for i in range(len(sentence))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]

def preprocess(file):
    sentences = []
    for line in open(file):
        sentence = []
        line = line.strip()
        line = re.sub(r"_+", "_", line)
        if not line:
            continue
        tokens = line.strip().split(" ")
        try:
            for token in tokens:
                if token.startswith("//"):
                    word = "/"
                    tag = token[2:]
                else:
                    word, tag = token.split("/")
                word = word.replace("_", " ")
                sentence.append((word, tag))
        except:
            continue
        sentences.append(sentence)
    return sentences

def raw_to_corpus():
    raw_folders = ["Trainset-POS-full", "Testset-POS"]
    data_folder = "data/vlsp2013/raw"
    train = []
    test = []
    for i, raw_folder in enumerate(raw_folders):
        files = listdir(join(data_folder, raw_folder))
        files = [join(data_folder, raw_folder, file) for file in files]
        for file in files:
            if i == 0:
                train += preprocess(file)
            else:
                test += preprocess(file)
    return train, test


def parse(path):
    sentences = []
    with open(path, "r", encoding="utf8") as f:
        sent = []
        for line in f:
            if line.strip():
                sent.append(line.strip().split())
            else:
                sentences.append(sent)
                sent = []
    sentences = preprocess(sentences)
    return sentences
if __name__ == "__main__":
    train, test = raw_to_corpus()
    print(train[10])
    crf = pickle.load(open("crf_pos.pkl", "rb"))
    # pickle.dump(train, open("train_pos.pkl", "wb"))
    # pickle.dump(test, open("test_pos.pkl", "wb"))
    # print("\tTrain size = {}, Test size = {}".format(len(train), len(test)))
    # X_train = [sent2features(s) for s in train]
    # y_train = [sent2labels(s) for s in train]

    X_test = [sent2features(s) for s in test]
    y_test = [sent2labels(s) for s in test]
    # crf = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     c1=0.05,
    #     c2=0.1,
    #     max_iterations=100,
    #     all_possible_transitions=True
    # )
    # crf.fit(X_train, y_train)
    # pickle.dump(crf, open("crf_pos.pkl", "wb"))
    labels = list(crf.classes_)
    y_pred = crf.predict(X_test)
    metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    labels.remove("Nc")
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    # sorted_labels = sorted(
    #     labels,
    #     key=lambda name: (name[1:], name[0])
    # )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))
    # print("=============================================")
    # for i in range(4):
    #     if i == 0:
    #         continue
    #     else:
    #         if i == 1:
    #             labels.remove("FW")
    #         elif i == 2:
    #             labels.remove("Z")
    #         else:
    #             labels.remove("I")
    #     sorted_labels = sorted(
    #         labels,
    #         key=lambda name: (name[1:], name[0])
    #     )
    #     print(metrics.flat_classification_report(
    #         y_test, y_pred, labels=sorted_labels, digits=3
    #     ))
    #     print("=============================================")