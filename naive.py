import numpy as np
import pandas as pd
import random
import os
from collections import deque
from numpy import array
import copy


# By Ian Hoole
# place the train, verify, and test files in a file labeled train in the same folder as the code

def printImage(image):
    # print function for debugging
    for i in image:
        line = ""
        for j in i:
            if j == 0:
                line += "@"
            else:
                line += " "
        print(line)


def findDA(arr, val):
    # helper function for the loop function
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == val:
                return (i, j)
    return (-1, -1)


def getBlWhImage(image):
    # Turns the image into black and white
    pixels = []
    for x in range(28):
        for y in range(28):
            if (int(image[x][y]) > 128):
                pixels.append(1)
            else:
                pixels.append(0)
    return np.reshape(pixels, (28, 28))


def symmetric(image):
    # function to compute the symmetry of the digit
    left = 27
    right = 0
    for i in image:
        tempLeft = 0
        tempRight = 27
        foundLeft = False
        foundRight = False
        while tempLeft < 28:
            if i[tempLeft] == 1:
                foundLeft = True
                break
            tempLeft += 1
        while tempRight >= 0:
            if i[tempRight] == 1:
                foundRight = True
                break
            tempRight -= 1
        if tempLeft < left and foundLeft:
            left = tempLeft
        if tempRight > right and foundRight:
            right = tempRight
    midpoint = (left + right) / 2
    count = 0
    total = 0
    for i in image:
        leftpoint = left
        rightpoint = right
        while leftpoint < midpoint and rightpoint > midpoint:
            total += 1
            if i[leftpoint] == i[rightpoint] and i[leftpoint] == 1:
                count += 1
            leftpoint += 1
            rightpoint -= 1
    return count / total


def verticalIntersections(image):
    # Gets the number of intersections in black and white image
    counts = []
    prev = 0
    for y in range(28):
        count = 0
        for x in range(28):
            current = int(image[x][y])
            if (prev != current):
                count += 1
            prev = current
        counts.append(count)
    average = sum(counts) / 28
    maximum = max(counts)
    return average, maximum


def horizontalIntersections(image):
    # Gets the number of horizontal intersections in black and white image
    counts = []
    prev = 0
    for y in range(28):
        count = 0
        for x in range(28):
            current = int(image[y][x])
            if (prev != current):
                count += 1
            prev = current
        counts.append(count)
    average = sum(counts) / 28
    maximum = max(counts)
    return average, maximum


def calculateDensity(image):
    # calculates the density
    count = 0
    for x in range(28):
        for y in range(28):
            count = count + int(image[x][y])
    return count / (28 * 28)


def spaceLength(image):
    # calculate how much empty space is between the
    totalCount = 0
    maxCount = 0
    for i in image:
        count = 0
        left = 0
        right = 27
        while left < 28 and i[left] != 1:
            left += 1
        while right > left and i[right] != 1:
            right -= 1
        while left < right:
            entered = True
            left += 1
            if i[left] == 0:
                count += 1
        totalCount += count
        if maxCount < count:
            maxCount = count
    return maxCount, totalCount / 28


def width(image):
    # calculate the width of the image
    width = 0
    for i in image:
        left = 0
        right = 27
        while left < 28 and i[left] != 1:
            left += 1
        while right > left and i[right] != 1:
            right -= 1
        width += right - left
    return width / 28


def findLoops(image):
    # find the number of loops in the number
    newImage = copy.deepcopy(image)
    queue = deque()
    count = -1
    node = findDA(newImage, 0)
    while node != (-1, -1):
        count += 1
        queue.append(node)
        newImage[node[0]][node[1]] = 1
        while len(queue) > 0:
            node = queue.popleft()
            if (node[0] > 0 and newImage[node[0] - 1][node[1]] == 0):
                newImage[node[0] - 1][node[1]] = 1
                queue.append((node[0] - 1, node[1]))
            if (node[0] < 27 and newImage[node[0] + 1][node[1]] == 0):
                newImage[node[0] + 1][node[1]] = 1
                queue.append((node[0] + 1, node[1]))
            if (node[1] > 0 and newImage[node[0]][node[1] - 1] == 0):
                newImage[node[0]][node[1] - 1] = 1
                queue.append((node[0], node[1] - 1))
            if (node[1] < 27 and newImage[node[0]][node[1] + 1] == 0):
                newImage[node[0]][node[1] + 1] = 1
                queue.append((node[0], node[1] + 1))
        node = findDA(newImage, 0)
    return count


def open_images(path):
    # Opens the csv files and extracts the images from them and returns them
    images = []
    data = pd.read_csv(path)
    headers = data.columns.values

    labels = data[headers[0]]
    labels = labels.values.tolist()

    pixels = data.drop(headers[0], axis=1)

    for i in range(0, data.shape[0]):
        row = pixels.iloc[i].to_numpy()
        grid = np.reshape(row, (28, 28))
        images.append(grid)
    return labels, images

def getFeatureSet(image):
    # get an array of features for an image
    set = []
    # density:0
    set.append(calculateDensity(image))
    # symmetry:1
    set.append(symmetric(image))

    vertical = verticalIntersections(image)
    # vertical max:2
    set.append(vertical[1])
    # vertical avg:3
    set.append(vertical[0])

    horizontal = horizontalIntersections(image)
    # horizontal max:4
    set.append(horizontal[1])
    # horizontal avg:5
    set.append(horizontal[0])

    # loops:6 (additional feature 1)
    set.append(findLoops(image))

    inside = spaceLength(image)
    # max space:7 (additional feature 2)
    set.append(inside[0])
    # avg space:8 (additional feature 3)
    set.append(inside[1])
    # avg width of the number:9 (additional feature 4)
    set.append(width(image))
    return set

def maketrainset():
    # creates a set of training data
    array = []
    for i in range(10):
        path = "train_and_valid/train" + str(i) + ".csv"
        images = open_images(path)
        for j in images[1]:
            array.append((i, getBlWhImage(j)))
    return array

def train2():
    masterSet = maketrainset()
    histogram = []
    for i in range(10):
        input = []
        for j in range(10):
            input.append([0,0])
        histogram.append(input)
    masterFeatureSet = []
    for i in masterSet:
        masterFeatureSet.append((i[0], getFeatureSet(i[1])))
    for i in masterFeatureSet:
        for j in range(10):
            histogram[i[0]][j][0] += i[1][j]
    for i in range(len(histogram)):
        for j in range(len(histogram[i])):
            histogram[i][j][0] = histogram[i][j][0]/999
    for i in masterFeatureSet:
        for j in range(10):
            histogram[i[0]][j][1] += (histogram[i[0]][j][0] - i[1][j]) * (histogram[i[0]][j][0] - i[1][j])
    for i in range(len(histogram)):
        for j in range(len(histogram[i])):
            histogram[i][j][1] = histogram[i][j][1] / 999
    return histogram

def makeVerifyFeatures():
    # used to create a set of features
    array = []
    for i in range(10):
        path = "train_and_valid/valid" + str(i) + ".csv"
        images = open_images(path)
        for j in images[1]:
            array.append((i, getBlWhImage(j)))
    featureSet = []
    for i in array:
        featureSet.append((i[0], getFeatureSet(i[1])))
    # print(featureSet[800][1])
    return featureSet

def makeVerifySet():
    # used to make a set for verification
    array = []
    for i in range(10):
        path = "train_and_valid/valid" + str(i) + ".csv"
        images = open_images(path)
        for j in images[1]:
            array.append((i, getBlWhImage(j)))
    return array

def naive(a, b, neuron):
    if a > b:
        a, b = b, a
    test = []
    path = "train_and_valid/valid" + str(a) + ".csv"
    images = open_images(path)
    for j in images[1]:
        test.append((a, (getFeatureSet(getBlWhImage(j)))))
    path = "train_and_valid/valid" + str(b) + ".csv"
    images = open_images(path)
    for j in images[1]:
        test.append((b, (getFeatureSet(getBlWhImage(j)))))
    count = 0
    correct = 0
    for i in test:
        aVal = 0
        bVal = 0
        for k in range(10):
            if (neuron[a][k][0] - neuron[a][k][1] * 2) < i[1][k] < (neuron[a][k][0] + neuron[a][k][1] * 2):
                aVal += 1
            if (neuron[b][k][0] - neuron[b][k][1] * 2) < i[1][k] < (neuron[b][k][0] + neuron[b][k][1] * 2):
                bVal += 1
        if aVal > bVal:
            if i[0] == a:
                correct += 1
        else:
            if i[0] == b:
                correct += 1
        count += 1
    return correct / count

def main():
    print("building histograms")
    values = train2()
    while True:
        tempString = input("by ian hoole, input first value, or -1 to quit: ")
        a = int(tempString)
        if a == -1:
            break
        tempString = input("input second value: ")
        b = int(tempString)
        print(values)
        print("naive Bayes on valid set values:", a, b, "success rate:", naive(a, b, values))
    print("goodbye")

if __name__ == "__main__":
    main()