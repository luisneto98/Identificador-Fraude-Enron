#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL",0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
### your code below
maior = 0
for i in data:
    if(i[0]> maior):
        maior = i[0]
value1 = 1000000.0
value2 = 5000000.0
for name in data_dict:
    if(data_dict[name]["salary"] != "NaN" and data_dict[name]["salary"] > value1 and data_dict[name]["bonus"] > value2):
        print name

