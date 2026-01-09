import numpy as np

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.classes.sort()
        features_num = X.shape[-1]

        self.mean = np.zeros((len(self.classes), features_num), dtype=np.float64)
        self.var = np.zeros((len(self.classes), features_num), dtype=np.float64)
        self.prior = np.zeros(len(self.classes), dtype=np.float64)

        for i, cls in enumerate(self.classes):
            by_class = X[y == cls]
            self.mean[i, :] = by_class.mean(axis=0)
            self.var[i, :] = by_class.var(axis=0)
            self.prior[i] = len(by_class) / X.shape[0] + 1e-9
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]

        return np.array(y_pred)
    
    def _predict(self, x):
        probs = []
        for cls in self.classes:
            lh = self._calc_likelihood(cls, x)
            p = np.sum(np.log(lh)) + np.log(self.prior[cls])
            probs.append(p)
        
        return self.classes[np.argmax(probs)]
    
    def _calc_likelihood(self, cls, X):
        d = np.sqrt(2 * np.pi * self.var[cls] + 1e-9)
        e = np.exp(-(X - self.mean[cls])**2 / (2 * self.var[cls] + 1e-9))

        return e/d + 1e-9
