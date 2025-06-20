#based on their income and spending score predict purchase power of customer prepare a cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans


#load the dataset into dataframe
df = pd.read_csv('Mall_Customers.csv')

X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

#k-means clustering means to cluster the data of similar kind,distance between the data points is calculated and then the data points are grouped together
#centers are the mean of the data points in a cluster,centroids are the center of the clusters
#we calculate the euclidean distance between the data points and the every centroids to assign the data points to the clusters
    # we use elbow method to find the optimal number of clusters, elbow method is calculated with inertia, inertia is the sum of squared distances of samples to their closest cluster center

wcss_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)#the init parameter is used to initialize the centroids, k-means++ is a method to initialize the centroids in a smart way
    kmeans.fit(X) # fit the model to the data,like training the model
    wcss_list.append(kmeans.inertia_)#model.inertia_ is the sum of squared distances of samples to their closest cluster center, inertia is used to calculate the elbow point
#plot the elbow method graph
#plt.plot(range(1, 11), wcss_list)#range(1, 11) is 1 to 10 , as 11 is excluded in range function
#plt.title('Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()

#we can see that the elbow point is at 5, so we will use 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=1)
y_predict = kmeans.fit_predict(X)  # fit the model to the data and predict the clusters
print(y_predict)

#lets plot the graph to visualize the clusters assignment of data points
X_array = X
plt.scatter(X_array[y_predict == 0, 0], X_array[y_predict == 0, 1], s=100, c='Green', label='Cluster 1')
plt.scatter(X_array[y_predict == 1, 0], X_array[y_predict == 1, 1], s=100, c='Red', label='Cluster 2')
plt.scatter(X_array[y_predict == 2, 0], X_array[y_predict == 2, 1], s=100, c='Blue', label='Cluster 3')
plt.scatter(X_array[y_predict == 3, 0], X_array[y_predict == 3, 1], s=100, c='Purple', label='Cluster 4')
plt.scatter(X_array[y_predict == 4, 0], X_array[y_predict == 4, 1], s=100, c='Orange', label='Cluster 5')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig("cluster_plot.png")
print("Plot saved as cluster_plot.png")
#save the model
joblib.dump(kmeans, 'kmeans_model.pkl')
print("Model saved as kmeans_model.pkl")
#load the model
#kmeans_model = joblib.load('kmeans_model.pkl')

#why should we save the model?
#we save the model so that we can use it later without training it again, it saves time and resources
#we can use the model to predict the clusters of new data points
