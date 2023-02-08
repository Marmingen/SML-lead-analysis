import numpy as np
from DecisionStump import DecisionStump

class AdaBoost():
    def __init__(self,n_clf=5):
        self.n_clf = n_clf


    def fit(self, X, Y):
        """

        """
        n_samples, n_features = X.shape

        # init weights
        w = np.full(n_samples, (1/n_samples))
        
        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()

            min_error = float('inf')
            for feature in range(n_features):
                X_column = X[:, feature]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    missclassified = w[y != predictions]
                    error = sum(missclassified)

                    if error > .5:
                        error = 1 - error
                        polarity = -1

                    if error < min_error:
                        min_error = error

                        clf.polarity = polarity
                        clf.threshold = threshold
                        clf.feature_index = feature
            EPS = 1e-10
            clf.alpha = .5*np.log((1-error)/(error+EPS))

            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w) # normalize

            self.clfs.append(clf)




    def predict(self, X):
        """

        """
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred




