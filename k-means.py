import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

train_file = 'train.csv'
test_file = 'test.csv'

train_dataset = pd.read_csv(train_file)
test_dataset = pd.read_csv(test_file)

test_dataset = test_dataset.drop('id', axis='columns')  #sbhnoyme to id sto test set

x_train, y_train = train_dataset.iloc[:,:-1], train_dataset.iloc[:,-1]

kmeans = KMeans(n_clusters=5, n_init=10)
kmeans.fit(x_train)