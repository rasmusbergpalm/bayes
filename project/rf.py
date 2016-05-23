import numpy as np
from sklearn.metrics import f1_score
from data import Data
from sklearn.ensemble import RandomForestClassifier

print "Loading data..."
data = Data(0.01)

train_X, train_S, train_R = data.load("0*", True)
train_S = np.argmax(train_S.toarray(), axis=1)
train_R = np.argmax(train_R.toarray(), axis=1)

val_X, val_S, val_R = data.load("ff*")
val_S = np.argmax(val_S.toarray(), axis=1)
val_R = np.argmax(val_R.toarray(), axis=1)

print "Fitting sender model..."
nb = RandomForestClassifier(n_estimators=32)
nb.fit(train_X, train_S)
pred_S = nb.predict(val_X)

print "Sender performance:"
print f1_score(val_S, pred_S, average='micro')
print f1_score(val_S, pred_S, average='macro')

print "Fitting receiver model..."
nb = RandomForestClassifier(n_estimators=32)
nb.fit(train_X, train_R)
pred_R = nb.predict(val_X)

print "Receiver performance:"
print f1_score(val_R, pred_R, average='micro')
print f1_score(val_R, pred_R, average='macro')