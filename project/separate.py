import numpy as np
from sklearn.metrics import f1_score
from data import Data
from models import MultinomialBayesianNaiveBayes

data = Data()
train_X, train_S, train_R = data.load("[0-9a-e]*", True)
val_X, val_S, val_R = data.load("f*")

val_r = np.argmax(val_R.toarray(), axis=1)
val_s = np.argmax(val_S.toarray(), axis=1)

nbr = MultinomialBayesianNaiveBayes(0, 0.01)
nbr.fit(train_X, train_R)
pred_r = nbr.predict(val_X)

nbs = MultinomialBayesianNaiveBayes(0, 0.01)
nbs.fit(train_X, train_S)
pred_s = nbs.predict(val_X)

print "Receiver:"
print f1_score(val_r, pred_r, average='micro')
print f1_score(val_r, pred_r, average='macro')
print f1_score(val_r, pred_r, average='weighted')
print "Sender:"
print f1_score(val_s, pred_s, average='micro')
print f1_score(val_s, pred_s, average='macro')
print f1_score(val_s, pred_s, average='weighted')