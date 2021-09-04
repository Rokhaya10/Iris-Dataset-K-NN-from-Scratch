# IMPLEMENT-K-NN-from-Scratch
Objective
IMPLEMENT K-NN from Scratch 
The test problem we will be using in this tutorial is the iris classification.

The problem is comprised of 150 observations of iris flowers from three different species. There are 4 measurements of given flowers: sepal length, sepal width, petal length, and petal width, all in the same unit of centimeters. The predicted attribute is the species, which is one of Setosa, Versicolor, or Virginica.

It is a standard dataset where the species is known for all instances. As such we can split the data into training and test datasets and use the results to evaluate our algorithm implementation. Good classification accuracy on this problem is above 90% correct, typically 96% or better.

Save the file in your current working directory with the file name “iris.data“.

This tutorial is broken down into the following steps:

Handle Data: Open the dataset from CSV and split it into test/train datasets.

Similarity: Calculate the distance between two data instances.

Neighbors: Locate k most similar data instances.

Response: Generate a response from a set of data instances.

Accuracy: Summarize the accuracy of predictions.

Main: Tie it all together.

1. Handle Data

The first thing we need to do is load our data file. 

import csv

with open('iris.data.txt', 'r') as csvfile:

lines = csv.reader(csvfile)

for row in lines :

print (', '.join(row))

Next we need to split the data into a training dataset 

import csv

import random

def loadDataset(filename, split, trainingSet=[] , testSet=[]):

with open(filename, 'r') as csvfile:

  lines = csv.reader(csvfile)

  dataset = list(lines)

  for x in range(len(dataset)-1):

    for y in range(4):

      dataset[x][y] = float(dataset[x][y])

    if random.random() < split:

      complete code

    else:

      complete code

We can test this function out with our iris dataset, as follows:

trainingSet=[]

testSet=[]

loadDataset('iris.data', 0.66, trainingSet, testSet)

print ('Train: ' + repr(len(trainingSet)))

print ('Test: ' + repr(len(testSet)) )

2. Similarity

To make predictions we need to calculate the similarity between any two given data instances. This is needed so that we can locate the k most similar data instances in the training dataset for a given member of the test dataset and in turn, make a prediction.

Given that all four flower measurements are numeric and have the same units, we can directly use the Euclidean distance measure. 

Additionally, we want to control which fields to include in the distance calculation. Specifically, we only want to include the first 4 attributes. One approach is to limit the Euclidean distance to a fixed length, ignoring the final dimension.

Putting all of this together, you have to define the euclidean distance

import math

def euclideanDistance(instance1, instance2, length):

      Complete the function

Note here that 

number of elements in the instance1 =number of elements in the instance2 

the length refers to the number of elements in the instance1 

We can test this function with some sample data, as follows:

data1 = [2, 2, 2, 'a']

data2 = [4, 4, 4, 'b']

distance = euclideanDistance(data1, data2, 3)

print 'Distance: ' + repr(distance)

3. Neighbors

Now that we have a similarity measure, we can use it to collect the k most similar instances for a given unseen instance.

This is a straightforward process of calculating the distance for all instances and selecting a subset with the smallest distance values.

Below is the getNeighbors function that returns k most similar neighbors from the training set for a given test instance (using the already defined euclideanDistance function)

import operator

def getNeighbors(trainingSet, testInstance, k):

distances = []

length = len(testInstance)-1

for x in range(len(trainingSet)):

dist = euclideanDistance(testInstance, trainingSet[x], length)

distances.append((trainingSet[x], dist))

distances.sort(key=operator.itemgetter(1))

neighbors = []

for x in range(k):

neighbors.append(distances[x][0])

return neighbors

We can test out this function as follows:

trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]

testInstance = [5, 5, 5]

k = 1

neighbors = getNeighbors(trainSet, testInstance, 1)

print(neighbors)

4. Response

Once we have located the most similar neighbors for a test instance, the next task is to devise a predicted response based on those neighbors.

We can do this by allowing each neighbor to vote for their class attribute, and take the majority vote as the prediction.

Below provides a function for getting the majority voted response from a number of neighbors. It assumes the class is the last attribute for each neighbor.

import operator

def getResponse(neighbors):

classVotes = {}

for x in range(len(neighbors)):

response = neighbors[x][ ? ] #complete with appropriate number

if response in classVotes:

Complete the if clause

sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

return sortedVotes[0][0]

We can test out this function with some test neighbors, as follows:

neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]

response = getResponse(neighbors)

print(response)

This approach returns one response in the case of a draw, but you could handle such cases in a specific way, such as returning no response or selecting an unbiased random response.

5. Accuracy

We have all of the pieces of the kNN algorithm in place. An important remaining concern is how to evaluate the accuracy of predictions.

An easy way to evaluate the accuracy of the model is to calculate a ratio of the total correct predictions out of all predictions made, called the classification accuracy.

Below is the getAccuracy function that sums the total correct predictions and returns the accuracy as a percentage of correct classifications.

def getAccuracy(testSet, predictions):

Complete the function

return (correct/float(len(testSet))) * 100.0

We can test this function with a test dataset and predictions, as follows:

testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]

predictions = ['a', 'a', 'a']

accuracy = getAccuracy(testSet, predictions)

print(accuracy)

6. Main

We now have all the elements of the algorithm you can put them all in one main function

7. Another distance metric

In this part, you are asked to define another distance metric instead of euclidean distance

 
