# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 18:39:53 2023

@author: dell
"""
# importing the data# 

import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv("D:\\data science python\\NEW DS ASSESSMENTS\\crime_data.csv")
df
df.info()
df.isnull().sum()
pd.set_option('display.max_columns', None)
df
df.shape

# EDA #

#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #

import seaborn as sns
import matplotlib.pyplot as plt
data = ['Murder','Assault','UrbanPop','Rape']
for column in data:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
#so basically we have seen the ouliers at once without doing everytime for each variable using seaborn#

"""removing the ouliers"""

# List of column names with continuous variables

continuous_columns = ['Murder','Assault','UrbanPop','Rape'] 
 
# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df.copy()
for df.cloumns in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker_Length = Q1 - 1.5 * IQR
    upper_whisker_Length = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker_Length) & (data_without_outliers[column]<= upper_whisker_Length)]
# Print the cleaned data without outliers

print(data_without_outliers)
df = data_without_outliers
print(df)
df.shape
df.info()

#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df.hist()
df.skew()
df.kurt()
df.describe()
sns.distplot(df["Murder"])
sns.distplot(df["Assault"])
sns.distplot(df["UrbanPop"])
sns.distplot(df["Rape"])

# understanding the relationships between all the four variables#

X = df.iloc[:,1:5].values
X.shape

import seaborn as sns
sns.pairplot(df, vars=['Murder', 'Assault', 'UrbanPop', 'Rape'])
plt.show()

"""# we can find Positive or Negative Relationships
#correlation
#outliers
#Histograms
#Outliers from the above code between all the four variables instead of doing scatter plot"""

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

"""Values close to 1 indicate a strong positive correlation.
Values close to -1 indicate a strong negative correlation.
Values close to 0 indicate a weak or no correlation"""

# transformation of the data #

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X


""" Hierarchial (Agglomerative Clustering) """
#forming a group using clusters

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',linkage = 'complete')
Y = cluster.fit_predict(SS_X)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

#construction of dendogram

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("CrimeData Dendograms")
dend = shc.dendrogram(shc.linkage(SS_X,method = 'complete'))



plt.figure(figsize=(10, 7))
plt.scatter(SS_X[:, 0], SS_X[:, 1], c=SS_X[:, 2], cmap='rainbow')
plt.xlabel('X-axis Label for Feature 1')
plt.ylabel('Y-axis Label for Feature 2')
plt.title('Scatter Plot with Rainbow Colormap')
plt.colorbar(label='Feature 3 Value')  # Add a colorbar to indicate Feature 3 values
plt.show()


plt.figure(figsize=(10, 7))
plt.scatter(SS_X[:, 0], SS_X[:, 1], c=SS_X[:, 3], cmap='rainbow')
plt.xlabel('X-axis Label for Feature 1')
plt.ylabel('Y-axis Label for Feature 2')
plt.title('Scatter Plot with Rainbow Colormap')
plt.colorbar(label='Feature 4 Value')  # Add a colorbar to indicate Feature 4 values
plt.show()

""" in the above code the scatterplot with a rainbow colormap is useful for visualizing how the feature Q3
vary across the points in the 2D space defined by feature1 and feature 2.it helps us to understand the rlationship or patterns 
between these three variables and next code with the feature 4 variable"""




""" performing k means on the same data"""

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)

Y = kmeans.fit_predict(SS_X)

Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

kmeans.inertia_

kresults = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_predict(X)
    kresults.append(kmeans.inertia_)
    
    
kresults

import matplotlib.pyplot as plt
plt.scatter(x=range(1,11),y=kresults)
plt.plot(range(1,11),kresults,color="red")
plt.show()

"""according to the elbow method we can get clarity on upto  which k value should be choosen"""
"""  at a certain stage ,we will able to see minimal drop of inertia values from major drop of inertial value.those minimal drop 
inertial k-stages can be neglected or ignored"""

""" here in this case we have a sequence of minimal drop of inertia values from k=5 onwards so we can neglect them and we will choose
the k value as 4 in this case"""

#  DBSCAN  #
""" for clustering analysis the # standardization# is must"""
 
from sklearn.cluster import DBSCAN
db = DBSCAN(eps = 1.0,min_samples = 2)
X = pd.DataFrame(SS_X)
db.fit(X)

Y = db.labels_
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

df

df_new = pd.concat([df,Y_new],axis=1)

df_final = df_new[df_new[0] != -1]
df_final.shape


df_noise = df_new[df_new[0] == -1]
df_noise

df_1 = df_new[df_new[0] == 1].value_counts()
df_1

df_0 = df_new[df_new[0] == 0].value_counts()
df_0

df_2 = df_new[df_new[0] == 2].value_counts()
df_2

df_3 = df_new[df_new[0] == 3].value_counts()
df_3

df_4 = df_new[df_new[0] == 4].value_counts()
df_4

"""db = DBSCAN(eps = 1.5,min_samples = 2)"""

""" so in ths DBSCAN we have mainly two parameters epsilon/radius and min points so we have to change these parameter values
in order to check where our data set contains less number of noise points. in this above scenario when i keep my epsilon/radius value as 1.0 and min points as 2
i am getting 5 cluster formations along with 11 noise points . """

""" on increasing the epsilon value all falls under one cluster and reducing the epsilon / radius value resulting in the increase of noise points
that should not be the way"""

""" basically in clustering analysis we dont have a target variable we are creating target variable by ourselves here
so after generating the target variable using the three clustering techniques we will again perform the supervised learning models on these values 
and then for which clustering formation techniques we are getting good accuracy score that model number of clustes will be choosen """

