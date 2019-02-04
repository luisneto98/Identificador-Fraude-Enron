#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                                test_size=0.3, random_state=42)


### it's all yours from here forward!  
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
print clf.score(features_test,labels_test)
pred  = clf.predict(features_test)
print (pred == 1).sum()
print len(features_test)
count = 0
for i in labels_test:
    if(i != 1.0):
        count = count + 1.
print count/len(labels_test)
print "precision score: ", precision_score(labels_test, pred)
print "recall score: ", recall_score(labels_test, pred)
prev = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
prev_true = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print confusion_matrix(prev_true,prev,labels=[0,1])
print "precision score (question): ", precision_score(prev_true,prev)
print "recall score (question): ", recall_score(prev_true,prev)

