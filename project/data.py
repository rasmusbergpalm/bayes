import glob
from sklearn.feature_extraction.text import CountVectorizer


class Data:
    def __init__(self, min_df):
        self.sv = CountVectorizer()
        self.rv = CountVectorizer()
        self.dv = CountVectorizer(max_df=1.0, min_df=min_df)

    def load(self, pattern, train=False):
        docs = list()
        receivers = list()
        senders = list()
        for path in glob.glob("data/" + pattern):
            with open(path + "/parties.txt") as f:
                senders.append(f.readline())
                receivers.append(f.readline())
            with open(path + "/words.csv") as f:
                f.readline()
                docs.append(' '.join([line.rstrip().split(',')[5] for line in f]))

        if train:
            self.dv.fit(docs)
            self.sv.fit(senders)
            self.rv.fit(receivers)

        D = self.dv.transform(docs)
        S = self.sv.transform(senders)
        R = self.rv.transform(receivers)

        return D, S, R
