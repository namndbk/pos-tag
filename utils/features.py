import re
import pickle
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


def word2features(sent, i, is_training, bi_grams, tri_grams):
    word = sent[i][0] if is_training else sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        #   'word[-3:]': word[-3:],
        #   'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1][0] if is_training else sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.bi_gram()': ' '.join([word1, word]).lower() in bi_grams,
        })
        if i > 1:
            word2 = sent[i - 2][0] if is_training else sent[i - 2]
            features.update({
                '-2:word.tri_gram()': ' '.join([word2, word1, word]).lower() in tri_grams,
            })
            #    else:
            #        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0] if is_training else sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.bi_gram()': ' '.join([word, word1]).lower() in bi_grams,
        })
    if i < len(sent) - 2:
        word2 = sent[i + 2][0] if is_training else sent[i + 2]
        features.update({
            '+2:word.tri_gram()': ' '.join([word, word1, word2]).lower() in tri_grams,
        })
            #    else:
            #        features['EOS'] = True

    return features


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
        'word_in_dict': is_unigram(word, uni_grams),
        'word.lower': word.lower(),
        'isTitle': word.istitle(),
        'isNumber': word.isdigit(),
        'isUpper': word.isupper(),
        # 'isCapWithPeriod': is_cap_with_period(word),
        # 'endsWithDigit': ends_with_digit(word),
        # 'containsHyphen': contains_hyphen(word),
        # 'isDate': is_date(word) or is_month_year(word),
        # 'isCode': is_code(word),
        # 'isName': is_name(word),
        # 'isMixCase': is_mix_case(word),
        # 'd&comma': digit_and_comma(word),
        # 'd&period': digit_and_period(word),
        # 'wordShape': get_word_shape(word),

        # 'isRange': is_range(word),
        # 'isRate': is_rate(word),
    }

    if i > 0:
        previous_word = sentence[i - 1][0]

        features.update({
            'w(-1)': previous_word,
            'w(-1).lower': previous_word.lower(),
            'isTitle(-1)': previous_word.istitle(),
            'isNumber(-1)': previous_word.isdigit(),
            # 'isCapWithPeriod(-1)': is_cap_with_period(previous_word),
            # 'isName(-1)': is_name(previous_word),
            # 'wordShape(-1)': get_word_shape(previous_word),
            'w(-1)+w(0)': previous_word + ' ' + word,
            # 'w(-1)+w(0)_in_dict': is_bigram(previous_word + ' ' + word, bi_grams)
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
            # 'isNumber(+1)': next_word.isdigit(),
            # 'isCapWithPeriod(+1)': is_cap_with_period(next_word),
            # 'isName(+1)': is_name(next_word),
            # 'wordShape(+1)': get_word_shape(next_word),
            # 'w(0)+w(+1)': word + ' ' + next_word,
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
    # uni_grams, bi_grams, tri_grams = load_dict(
    #     "/content/drive/MyDrive/Colab Notebooks/ĐATN/data/dictionary/pyvi_dictionary.txt")
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
        words = []
        for token in tokens:
            if token.startswith("//"):
                word = "/"
            else:
                word = token.split("/")[0]
            words.append(word)
        for word in words:
            syllabels = word.split("_")
            syllabels = [item for item in syllabels if item]
            for i, syllabel in enumerate(syllabels):
                label = "B-W" if i == 0 else "I-W"
                sentence.append([syllabel, label])
        sentences.append(sentence)
    return sentences


def preprocess_vlsp2016(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        sent = []
        for line in f:
            if line.strip():
                line = line.strip().split("\t")
                tokens = line[0]
                tokens = tokens.strip().split("_")
                tokens = [token for i, token in enumerate(tokens)]
                output = [[token, "B-W"] if i == 0 else [token, "I-W"]
                          for i, token in enumerate(tokens) if token != ""]
                sent += output
            else:
                sentences.append(sent)
                sent = []
    return sentences


def main():
    # sents_1 = preprocess("/content/drive/MyDrive/Colab Notebooks/ĐATN/data/dataset/vtb.txt")
    # sents_2 = preprocess_vlsp2016("/content/drive/MyDrive/Colab Notebooks/ĐATN/data/vlsp2016/raw/train.txt")
    # for sent in sents_2:
    #     for tokens in sent:
    #         if tokens[0].strip() == "":
    #             print(tokens)
    # train, test = train_test_split(sents_1, test_size=0.33, random_state=2020)
    # train += sents_2 + preprocess_vlsp2016("/content/drive/MyDrive/Colab Notebooks/ĐATN/data/vlsp2016/raw/dev.txt")
    # test += preprocess_vlsp2016("/content/drive/MyDrive/Colab Notebooks/ĐATN/data/vlsp2016/raw/test.txt")
    train, test = raw_to_corpus()
    print("VLSP")
    print("\tTrain size = {}, Test size = {}".format(len(train), len(test)))
    # X_train = [sent2features(s) for s in train]
    # y_train = [sent2labels(s) for s in train]

    X_test = [sent2features(s) for s in test]
    y_test = [sent2labels(s) for s in test]
    # crf = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     c1=0.1,
    #     c2=0.1,
    #     max_iterations=100,
    #     all_possible_transitions=True
    # )
    # crf.fit(X_train, y_train)
    # pickle.dump(crf, open("crf_wg.pkl", "wb"))
    crf = pickle.load(open("crf_ws.pkl", "rb"))
    labels = list(crf.classes_)
    y_pred = crf.predict(X_test)
    metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))


if __name__ == '__main__':
    main()
