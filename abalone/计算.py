# a= 0.0000065
# b=0.58
# c=a/b
# print(c)
#
# c = float(a/b)
#
# # Display the result
# print(c)
#
# height = 10/0.000011
# print(height,"m")

import matplotlib.pyplot as plt
from matplotlib import style
# style.use('ggplot')
import numpy as np


class K_Means:
    def __init__(self, k=3, tol=0.01, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        self.centroids = {}

    def fit(self, data):

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            print(i, self.centroids, ' centroids')
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            print(self.classifications, i, ' * ')

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in
                             self.centroids]  # check distance of every data point to every centroid
                print(distances, featureset, ' dist - featureset')
                classification = distances.index(min(distances))  # get index of centroid that best suits the data point
                self.classifications[classification].append(featureset)  # assign the centroid id to the data point

            print(self.classifications, i, ' + ')

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:  # update centroid position
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                break
        return self.centroids

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# main

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [1, 3],
              [8, 9],
              [0, 3],
              [5, 4],
              [6, 4], ])

plt.scatter(X[:, 0], X[:, 1], s=150)
plt.savefig('data.png')
print(X, ' data')

colors = 10 * ["g", "r", "c", "b", "k"]
clf = K_Means()
x = clf.fit(X)
print(x, 'final centroids')

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

# in case we wish to test if data belongs to a cluster

##unknowns = np.array([[1,3],
##                     [8,9],
##                     [0,3],
##                     [5,4],
##                     [6,4],])
##
##for unknown in unknowns:
##    classification = clf.predict(unknown)
##    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)
##

plt.savefig('kmeans.png')