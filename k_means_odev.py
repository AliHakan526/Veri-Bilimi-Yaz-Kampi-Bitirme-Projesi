

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt





data = pd.read_csv('dava.csv')


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def ElbowCheck(string):
    X = data[[string]].values
    wcv = []  
    K_range = range(1, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(X)
        wcv.append(kmeans.inertia_)  

    plt.plot(K_range, wcv, 'bo-')
    plt.xlabel("Küme Sayısı (k)")
    plt.ylabel("Küme İçi Hata (WCV)")
    plt.title(f"Elbow Yöntemi ({string})")
    plt.show()

def KmeansAlgo(optimal_k, string):
    X = data[[string]].values  
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
    data["cluster"] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    print(data)
    print("Küme Merkezleri:\n", centroids)
    plt.figure(figsize=(8,5))
    for cluster in range(optimal_k):
        plt.scatter(data[data["cluster"]==cluster][string], 
                    [0]*len(data[data["cluster"]==cluster]),  
                    label=f'Cluster {cluster}')
    plt.scatter(centroids, [0]*len(centroids), color='black', marker='x', s=100, label='Centroids')
    plt.xlabel(string)
    plt.title(f'K-Means Clustering ({string})')
    plt.legend()
    plt.show()



KmeansAlgo(4,"Legal Fees (USD)")