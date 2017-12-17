import math
import operator
import numpy as np

# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# distances: https://numerics.mathdotnet.com/distance.html


# Calculating similarity
def euclideanDistance(instance1, instance2):
    distance = 0
    if len(instance1) == len(instance2):
        feature_number = len(instance1)
        for x in range(feature_number):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    else :
        print("Instance1 and Instance2 have different number of features")


def getNeighbours(XtrainingSet, ytrainingSet, xtestInstance, k):
    distances = []
    for x in range(len(XtrainingSet)):
        dist = euclideanDistance(xtestInstance, XtrainingSet[x])
        distances.append((XtrainingSet[x], dist, ytrainingSet[x]))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors


# Get prediction
def getPrediction(neighbors, weighted_prediction=True):
    if weighted_prediction is True:
        counter = 0
        denominator = 0
        for neighbour in neighbors:
            counter += 1/neighbour[-2]*neighbour[-1]
            denominator += 1/neighbour[-2]
        return counter/denominator
    else:
        neighbors_predictions = [neighbour[-1] for neighbour in neighbors]
        return np.mean(neighbors_predictions)


if __name__ == "__main__":

    # Eucidean distance test
    # data1 = [2, 2, 2]
    # data2 = [2, 5, 2]
    # distance = euclideanDistance(data1, data2)
    # print('Distance: ' + repr(distance))

    # getting Neighbours test
    trainSet = [[2, 2, 2], [4, 4, 4]]
    ySet = [0, 1]
    testInstance = [5, 5, 5]

    neighbors = getNeighbours(XtrainingSet=trainSet, xtestInstance=testInstance, ytrainingSet=ySet, k=2)
    print(neighbors)

    print(getPrediction(neighbors, weighted_prediction=True))


