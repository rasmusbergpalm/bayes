import numpy as np
import glob
from sklearn.feature_extraction.text import CountVectorizer

class Data:
    def __init__(self):
        self.sv = CountVectorizer()
        self.rv = CountVectorizer()
        self.dv = CountVectorizer(max_df=1.0, min_df=0.01)

    def load(self, pattern, train=False):
        docs = list()
        receivers = list()
        senders = list()
        for path in glob.glob("data/" + pattern):
            with open(path) as f:
                receivers.append(f.readline())
                senders.append(f.readline())
                docs.append(f.readline())

        if train:
            self.dv.fit(docs)
            self.sv.fit(senders)
            self.rv.fit(receivers)

        D = self.dv.transform(docs)
        S = self.sv.transform(senders)
        R = self.rv.transform(receivers)

        return D, S, R
