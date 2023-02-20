### IMPORTS ###
import os
import sys
import numpy as np

# Check folders so it works for different OS:s
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting/mains", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))

from DataPreparation import DataPreparation
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


#gridsearch
# kolla main i max

### MAIN ###

def main():
    # Get the data sets
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True, gender = False, normalize = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Use SMOTE for over sampling
    sm = SMOTE(random_state = 42)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))

    # ML model
    GDB = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, \
            max_depth = 1, random_state = 0)
    model = GDB.fit(X_train, Y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    target_names = ['Female', 'Male']
    
    # Print statistics
    report = classification_report(Y_test, y_pred, target_names = target_names)
    print("REPORT")
    print(report)

if __name__ == "__main__":
    main()
