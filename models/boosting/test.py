from sklearn import datasets

import os
import sys
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))
from PCA import PCA
data = datasets.load_iris()
x = data.data
y = data.target

pca = PCA(2)
pca.fit(x)
x = pca.transform(x)


