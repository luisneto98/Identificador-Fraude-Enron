#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
count = 0
maior = 0
import math
for value in enron_data:
    #if(enron_data[value]["poi"] == 1):
    #    count  = count + 1
    #if(enron_data[value]['total_payments'] > maior and (value.find('LAY') != -1 or value.find('SKILLING') != -1 or value.find('FASTOW') != -1)):
    #    name = value
    #    maior = enron_data[value]['total_payments']
    #if(enron_data[value]['email_address']  != "NaN"):
    #    count =count +1
    if(enron_data[value]["poi"] == 1 and enron_data[value]['total_payments']  == "NaN"):
        count = count + 1
print(count)
print(len(enron_data))
