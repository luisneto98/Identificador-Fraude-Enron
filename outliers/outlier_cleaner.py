#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    from math import floor
    cleaned_data = []
    diffs = []
    
    for key in range(len(predictions)):
        diffs.append((abs(predictions[key]-net_worths[key])[0],key))
    def getKey(item):
        return item[0]
    diffs = sorted(diffs, key=getKey)
    for i in range(0,int(len(diffs)*0.90)):
        key = diffs[i][1]
        cleaned_data.append((ages[key][0],net_worths[key][0],diffs[key][0]))
    ### your code goes here

    
    return cleaned_data

