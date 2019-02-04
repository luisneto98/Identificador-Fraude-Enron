#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import matplotlib.pyplot
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
poi = 0
npoi = 0
print len(data_dict[list(data_dict)[0]].keys())
for name in data_dict:
    if(data_dict[name]['poi'] == 1):
        poi = poi + 1
    else:
        npoi = npoi+1
print "poi:",poi
print "npoi",npoi
feats = {}
for feat in data_dict[list(data_dict)[0]].keys():
    feats[feat] = 0
for name in data_dict:
    for feat in feats:
        if(data_dict[name][feat] == "NaN"):
            feats[feat] = feats[feat] + 1
print feats

### Task 1: Select what features you'll use.
ini_feature_list = ['poi','salary', 'to_messages', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'from_messages', 'total_stock_value', 'expenses', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'from_poi_to_this_person']
# Separandos dados para realizar testes de melhores features
data = featureFormat(data_dict,ini_feature_list , sort_keys = True)
labels, features = targetFeatureSplit(data)

# criando arvore de decisao para obter as melhores features
# ( todas as que tiverem importancida de pelo menos 10% )
clf  = DecisionTreeClassifier(random_state=2)
clf = clf.fit(features,labels)

# lista que ficara apenas com as features mais importantes
new_feature_list = ini_feature_list[1:]

# retirando features que possuem menos que 10% de importancia
for i in range(len(clf.feature_importances_)):
    if(clf.feature_importances_[i] <= np.percentile(clf.feature_importances_,90)):
        new_feature_list.remove(ini_feature_list[i+1])
#    print  clf.feature_importances_[i], ini_feature_list[i+1]
print "lista de features selecionadas: ",new_feature_list #['total_payments', 'bonus', 'restricted_stock', 'expenses']
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] + new_feature_list# You will need to use more features
#features_list = ['poi', 'total_payments', 'bonus', 'restricted_stock', 'expenses']


### Task 2: Remove outliers
def nan_for_0(value):
    if(value == "NaN"):
        return 0
    return value
to_remove = []
print "outlier(s): "
for name in data_dict:
    if(nan_for_0(data_dict[name]['expenses']) > 5000000 ):
        print name
data_dict.pop("TOTAL")
for name in data_dict:
    expenses = nan_for_0(data_dict[name]['expenses'])
    bonus = nan_for_0(data_dict[name]['bonus'])
#    matplotlib.pyplot.scatter( expenses, bonus )

#matplotlib.pyplot.xlabel("expenses")
#matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()
### Task 3: Create new feature(s)
shared_receipt_with_poi_max = 0
shared_receipt_with_poi_min = 10000000000 ## infinity
    
def get_division(num1,num2):
    if(num1 == 'NaN'):
        num1 = 0
    if(num2 == 'NaN'):
        num2 = 0
    if(num2 == 0):
        return 0
    return num1/num2
            
def get_relation(to_poi,from_poi,shared_receipt_with_poi):
    return nan_for_0(to_poi)+nan_for_0(from_poi)+nan_for_0(shared_receipt_with_poi)

for name in data_dict:
    data_dict[name]["relationship_with_poi"] = get_relation(data_dict[name]["from_this_person_to_poi"],data_dict[name]["from_poi_to_this_person"],data_dict[name]["shared_receipt_with_poi"])

#features_list.append("relationship_with_poi")# adicionando nova feature a lista

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#for SVM
min_max = preprocessing.MinMaxScaler()
data = min_max.fit_transform(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn import svm
parameters_for_gnd = {}
gnb = GaussianNB()
parameters_for_dct = {'min_samples_split':range(2,20),'max_depth':range(3,100)}#,'max_features':range(1,5),'max_leaf_nodes':[10,50,100,200,500]}
dtc = DecisionTreeClassifier()
parameters_for_svm = {'kernel':['rbf'], 'C':[1, 10,100,1000,10000],'gamma':[0.005,0.05,0.1,0.2]}
svr = svm.SVC()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## gaussianGN
gsc_gnb = GridSearchCV(gnb,parameters_for_gnd,cv=2)
gsc_gnb.fit(features_train, labels_train)
accr_gnb = gsc_gnb.best_score_
#gnb.set_params(**gsc_gnb.best_params_)
print "GAUSSIAN: ",accr_gnb


##decision tree
gsc_dtc = GridSearchCV(dtc,parameters_for_dct,cv=2)
gsc_dtc.fit(features_train, labels_train)
accr_dtc = gsc_dtc.best_score_
#dtc.set_params(**gsc_dtc.best_params_)
print "DECISION TREE: ",accr_dtc

##SVM
gsc_svr = GridSearchCV(svr, parameters_for_svm,cv=2)
gsc_svr.fit(features, labels)
accr_svr = gsc_svr.best_score_
#svr.set_params(**gsc_svr.best_params_)
print "SVM: ",accr_svr

#clf = gsc_gnb.best_estimator_
print"gnb"

clf = gsc_dtc.best_estimator_
print "dtc"

#clf = gsc_svr.best_estimator_
print "svr"
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)