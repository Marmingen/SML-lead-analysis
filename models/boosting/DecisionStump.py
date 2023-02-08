import numpy as np

class DecisionStump():
    def __init__(self):
        self.polarity = 1 # classified as 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        """
        X: sample
        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions

