import sys
import os
import codecs
import pickle
import re
import string
import unicodedata as ud
import sklearn_crfsuite
from utils.features import sent2features
import preprocess_pos_vlsp2016
from pyvi import ViPosTagger
from underthesea import word_tokenize, pos_tag



def sylabelize(text):
    text = ud.normalize('NFC', text)

    specials = ["==>", "->", "\.\.\.", ">>"]
    digit = "\d+([\.,_]\d+)+"
    email = "([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
    # web = "^(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$"
    web = "\w+://[^\s]+"
    # datetime = [
    #    "\d{1,2}\/\d{1,2}(\/\d{1,4})(^\dw. )+",
    #    "\d{1,2}-\d{1,2}(-\d+)?",
    # ]
    word = "\w+"
    non_word = "[^\w\s]"
    abbreviations = [
        "[A-ZĐ]+\.",
        "Tp\.",
        "Mr\.", "Mrs\.", "Ms\.",
        "Dr\.", "ThS\."
    ]

    patterns = []
    patterns.extend(abbreviations)
    patterns.extend(specials)
    patterns.extend([web, email])
     # patterns.extend(datetime)
    patterns.extend([digit, non_word, word])

    patterns = "(" + "|".join(patterns) + ")"
    if sys.version_info < (3, 0):
        patterns = patterns.decode('utf-8')
    tokens = re.findall(patterns, text, re.UNICODE)

    return text, [token[0] for token in tokens]


def tokenize(text):
    text, tokens = sylabelize(text)
    new_tokens = []
    for token in tokens:
        new_tokens.append((token, "U"))
    crf = pickle.load(open("crf_wg.pkl", "rb"))
    X = [sent2features(s) for s in [new_tokens]]
    y_pred = crf.predict(X)
    y_pred = y_pred[-1]
    output = []
    token = ""
    for w, t in zip(tokens, y_pred):
        if t == "B-W":
            if token == "":
                token += w
            else:
                output.append(token)
                token = w
        elif t == "I-W":
            token += " " + w
    if token != "":
        output.append(token)
    return output


def pos_tag(text):
    tokens = tokenize(text)
    crf_pos = pickle.load(open("crf_pos.pkl", "rb"))
    new_tokens = []
    for token in tokens:
        new_tokens.append((token, "U"))
    X = [preprocess_pos_vlsp2016.sent2features(s) for s in [new_tokens]]
    y_pred = crf_pos.predict(X)
    out = []
    for w, t in zip(tokens, y_pred[-1]):
        out.append((w, t))
    return out
    


if __name__ == "__main__":
    text = 'Đại học Sư phạm TP HCM thành lập năm 1976, tiền thân là Đại học Sư phạm Quốc gia Sài Gòn ra đời năm 1957, hiện là một trong hai trường sư phạm trọng điểm của cả nước.'
    txt = "Chúng tôi là sinh viên đại học Bách Khoa Hà Nội."
    # text, tokens = sylabelize(text)
    # crf = pickle.load(open("crf_wg.pkl", "rb"))
    # X = [sent2features(s) for s in [tokens]]
    # y_pred = crf.predict(X)
    # print(y_pred)
    print(pos_tag(text))
    print(pos_tag(text))