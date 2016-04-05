import numpy as np
from scipy.special import gammaln
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import lil_matrix, csr_matrix, find


def betaln(x, axis):
    return np.sum(gammaln(x), axis=axis) - gammaln(np.sum(x, axis=axis))


class CombinedMultinomialBayesianNaiveBayes:
    def __init__(self, eta, alpha, beta):
        self.eta = eta
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, R, S):
        """
        X: sparse n-samples x n-words
        R: sparse (one hot) n-samples x n-receivers
        S: sparse (one hot) n-samples x n-senders
        """
        self.n_receivers = R.shape[1]
        self.n_senders = S.shape[1]
        self.n_words = X.shape[1]

        self.m = R.sum(axis=0).A.T  # n-receivers x 1
        self.c = safe_sparse_dot(R.T, S)  # n-receivers x n-senders

        self.lrp = np.log(self.eta + self.m)  # n-receivers x 1
        k = self.alpha + self.c.toarray()
        self.lrsp = np.log(k) - np.log(np.sum(k, axis=1, keepdims=True))  # n-receivers x n-senders

        receivers = np.argmax(R.toarray(), axis=1)
        senders = np.argmax(S.toarray(), axis=1)

        self.W = list()  # sparse n-receivers x n-senders x n-words
        for i in range(self.n_receivers):
            self.W.append(lil_matrix((self.n_senders, self.n_words)))

        for i in range(X.shape[0]):
            self.W[receivers[i]][senders[i], :] += X[i, :]

        for i in range(self.n_receivers):
            self.W[i] = csr_matrix(self.W[i])

    def predict_log_proba_sender(self, x, r):
        """
        x: sparse 1 x n-features
        :returns 1 x n-senders vector of log probabilities
        """

        _, j, xj = find(x)

        wj = self.W[r][:, j].toarray() + self.beta
        a = betaln(wj + xj, axis=1)
        b = betaln(wj, axis=1)
        lp = self.lrp[r] + self.lrsp[r] + a - b

        return lp

    def predict_sender(self, x, r):
        """
        x: sparse n-samples x n-features
        :returns two lists, of receivers and senders respectively
        """
        senders = list()
        n_docs = x.shape[0]
        for i in range(n_docs):
            lp = self.predict_log_proba_sender(x[i], r[i])
            s = np.argmax(lp)
            senders.append(s)

        return np.array(senders)


class MultinomialBayesianNaiveBayes:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def betaln(self, x, axis):
        return np.sum(gammaln(x), axis=axis) - gammaln(np.sum(x, axis=axis))

    def fit(self, X, Y):
        """
        X: sparse n-samples x n-features
        Y: sparse n-samples x n-classes
        """

        self.W = safe_sparse_dot(Y.T, X)  # n-classes x n-features
        self.C = Y.sum(axis=0).A  # 1 x n-classes

        self.lprior = np.log(self.C + self.alpha)

    def predict_log_proba(self, x):
        """
        x: sparse n-samples x n-features
        """
        lp = np.zeros((x.shape[0], self.C.size))
        for i in range(x.shape[0]):
            _, j, xj = find(x[i, :])
            wj = self.W[:, j].toarray() + self.beta
            lp[i, :] = self.lprior + betaln(wj + xj, axis=1) - betaln(wj, axis=1)

        return lp

    def predict(self, x):
        """
        x: sparse n-samples x n-features
        """
        return np.argmax(self.predict_log_proba(x), axis=1)

    def predict_proba(self, x):
        """
        x: sparse n-samples x n-features
        """
        lp = self.predict_log_proba(x)
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)
