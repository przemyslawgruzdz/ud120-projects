#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)



# the features to be used
features_list = ['poi', 'salary', 'exercised_stock_options']

def finance_kmeans(data_dict, features_list):
    data = featureFormat(data_dict, features_list )
    poi, finance_features = targetFeatureSplit( data )

    # plot the first 2 features
    for f in finance_features:
        plt.scatter( f[0], f[1] )

    # k-means clustering
    from sklearn.cluster import KMeans
    clf = KMeans(2)
    clf.fit(finance_features)
    pred = clf.predict(finance_features)

    # show the clustering
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=features_list[1], f2_name=features_list[2])
    
    return None

finance_kmeans(data_dict, features_list)
features_list = ['poi', 'salary', 'exercised_stock_options', 'total_payments']
finance_kmeans(data_dict, features_list)

import pandas as pd

df = pd.DataFrame(data_dict)
df.loc['exercised_stock_options',:] = pd.to_numeric(df.loc['exercised_stock_options',:], errors='coerce')
print df.loc['exercised_stock_options',:].max(skipna=True)
print df.loc['exercised_stock_options',:].min(skipna=True)

df.loc['salary',:] = pd.to_numeric(df.loc['salary',:], errors='coerce')
print df.loc['salary',:].max(skipna=True)
print df.loc['salary',:].min(skipna=True)

import numpy as np
from sklearn.preprocessing import MinMaxScaler

features_list = ['poi', 'salary', 'exercised_stock_options']

data = featureFormat(data_dict, features_list)
_, salary, stock = zip(*data)

# put the features into 2-D numpy arrays
salary = np.array(salary).reshape((len(salary),1))
stock = np.array(stock).reshape((len(stock),1))

# rescale
scaler = MinMaxScaler()
salary = scaler.fit_transform(salary)
print '$200,000 becomes {0}'.format(scaler.transform([[200000.]])[0][0])

stock = scaler.fit_transform(stock)
print '$1,000,000 becomes {0}'.format(scaler.transform([[1000000.]])[0][0])
