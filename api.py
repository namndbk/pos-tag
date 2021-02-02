from underthesea import word_tokenize
from syllable import tokenize, pos_tag
# from pos_tag.tag import pos_tag


from flask import Flask, request, render_template
from flask import json, jsonify


app = Flask(__name__)


@app.route("/tokenize", methods=["GET","POST"])
@app.route("/tokenize/crf", methods=["GET","POST"])
def tokenize_api():
    if request.method == "GET":
        message = {
            "text": "Hello"
        }
    else:
        doc = request.get_json()["text"]
        out = tokenize(doc.strip())
        output = ""
        for token in out:
            output += "_".join(x for x in token.split()) + " "
        message = {
            "text": output.strip()
        }
    return jsonify(message)


@app.route("/pos", methods=["GET","POST"])
@app.route("/pos/bilstm", methods=["GET","POST"])
def pos_api():
    if request.method == "GET":
        message = {
            "text": "Hello"
        }
    else:
        doc = request.get_json()["text"]
        out = pos_tag(doc.strip())
        output = ""
        for token in out:
            output += "_".join(x for x in token[0].split()) + "/" + token[-1] + " "
        message = {
            "text": output.strip()
        }
    return jsonify(message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
