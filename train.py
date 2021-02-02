import lsmt

import pickle
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import dataset
from vocab import Vocab


with open("test.pkl", "rb") as f:
    test = pickle.load(f)
with open("train.pkl", "rb") as f:
    train = pickle.load(f)

X_test, y_test = dataset.create_word_data(test)
X_train, y_train = dataset.create_word_data(train)
model = lsmt.create_model()


his = model.fit(X_train, y_train, batch_size=128, epochs=35, validation_split=0.1)

pickle.dump(model, open("v1.pkl", "wb"))

scores = model.evaluate(X_test, y_test)
print(f"{model.metrics_names[1]}: {scores[1] * 100}")
y_pred = model.predict(X_test)

label = [0, 1]
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))
