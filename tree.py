import numpy as np
import pandas as pd
import math
import os
import copy
import random
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def unit_step_func(x):
    return np.where(x > 0 , 1, 0)

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


def verticalIntersections(image):
    # Gets the number of vertical intersections in black and white image
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
    average = sum(counts)/28
    maximum = max(counts)
    return average/maximum

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
    average = sum(counts)/28
    maximum = max(counts)
    return average/maximum


def calculateDensity(image):
    # calculates the density
    count = 0
    for x in range(28):
        for y in range(28):
            count = count + int(image[x][y])
    return count/(28*28)


def nextTarget(image):
    #returns the first value in the matrix that is black
    for x in range(28):
        for y in range(28):
            if int(image[x][y]) == 0:
                return x, y
    return -1, -1


def floodFillWhite(image, x, y):
    #flood fills image with white from given x and y
    if (x <= 27 and x >= 0) and (y <= 27 and y >= 0) and image[x][y] == 0:
        image[x][y] = 1
        floodFillWhite(image, x+1, y)
        floodFillWhite(image, x-1, y)
        floodFillWhite(image, x, y+1)
        floodFillWhite(image, x, y-1)
    return image



def checkLoops(image):
    copyI = copy.deepcopy(image)
    loops = 0

    target = nextTarget(copyI)
    if target[0] == -1:
        #this should only fire if the whole image starts out white
        return loops
    copyI = floodFillWhite(copyI, target[0], target[1])

    target = nextTarget(copyI)
    while(target[0] != -1):
        #every time this loop runs we have detected an additional loop
        copyI = floodFillWhite(copyI, target[0], target[1])
        loops += 1
        target = nextTarget(copyI)
    return loops
 

def widthHeight(image):
    width = 0
    hieght = 0
    wtemp = 0
    htemp = 0
    for y in range(28):
        for x in range(28):
            if image[y][x] == 1:
                wtemp += 1
                htemp += 1
                #break
        if wtemp > width:#update width
            width = wtemp
        if htemp > 0:
          hieght += 1
        wtemp = 0
        htemp = 0
      
    return hieght/width

def leftRightDensity(image):
    # left right density comparison
    left = 0
    right = 0
    for x in range(28):
        for y in range(28):
          if x < 14:
              left += int(image[x][y])
          else:
              right += int(image[x][y])
    return left/right

def upDownDensity(image):
    # left right density comparison
    up = 0
    down = 0
    for x in range(28):
        for y in range(28):
          if y < 13:
              up += int(image[x][y])
          else:
              down += int(image[x][y])
    return up/down

def diagonal(image):#Euclidean distance
    posdiag = []
    negdiag = []
    index1 = [] 
    index2 = []

    for row in range(28):#gets index of first 1 encountered 
        if negdiag != []: 
            break
        for col in range(28):
            if negdiag != []: 
                break
            if image[row][col] == 1:
                negdiag.append([row,col])#1st point for negdiag

    for row in range(28):#gets last column of first 1 encountered 
        if posdiag != []: 
            break
        for col in range(28):
            if posdiag != []: 
                break
            if image[27-row][col] == 1:
                posdiag.append([27-row,col])#first point for posdiag

    for col in range(28):
      if index1 != []:
          break
      if image[negdiag[0][0]][27-col] == 1:
          index1.append([negdiag[0][0], 27-col])#second point for posdiag

    for col in range(28):
      if index2 != []:
          break
      if image[posdiag[0][0]][27-col] == 1:
          index2.append([posdiag[0][0], 27-col])#second point for negdiag

#works for python 3.9+
    ePosDistance = math.dist([posdiag[0][0], posdiag[0][1]], [index1[0][0], index1[0][1]])#Euclidean distance
    eNegDistance = math.dist([negdiag[0][0], negdiag[0][1]], [index2[0][0], index2[0][1]])

    return ePosDistance/eNegDistance

def symmetryXor(image):
    halfImage = []
    output = []
    for x in range(14):#copy half of image
        halfImage.append(image[x][0:14])
    
    for x in range(14):
        for y in range(14):
            if halfImage[x][y] != image[x][27-y]:#!= is equivalent to xor
                output.append(1)#1,0 or 0,1 were compared
            else:
              output.append(0)#0,0 1,1

    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    result = len( get_indexes(0,output) )#amount of pixels are symmetrical

    return result/28

def fullfil(image):#make sure we send a array of 28x28(single image) or rearrange the arrays used
    digit = []

    for y in range(28):
        temp = 0
        for x in range(28):
            if image[y][x] == 1:
                temp += 1
        digit.append(temp)

    vmax = max(digit)
    vlen = len(digit)
    vsum = sum(digit)

    return abs( (vlen*vmax)-vsum )

def setupData(dataSet, dataLabel):
    train = []
    labels = []
    for i in range(len(dataSet)):
        for j in range(len(dataSet[i])):
            features = []
            digit = dataSet[i][j]#single digit

            features.append(verticalIntersections( digit ) )
            features.append(horizontalIntersections( digit ) )
            features.append(calculateDensity( digit ) )
            features.append(checkLoops( digit ) )
            features.append(leftRightDensity( digit ) )
            features.append(upDownDensity( digit ) )
            features.append(symmetryXor( digit ) )
            features.append(widthHeight( digit ) )
            features.append(diagonal( digit ) )
            features.append(fullfil( digit ) )
            labels.append(dataLabel[i][j])
            #features.append( -1 )#bias
            #if dataLabel != -1:
                #features.append( dataLabel[i][j] )#label

            train.append( features )
    temp = list(zip(train, labels))
    random.shuffle(temp)
    train, labels = zip(*temp)
    train, labels = list(train), list(labels)
    return train, labels



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
        images.append(getBlWhImage(grid))
    return labels, images

def open_test(path):
    # Opens the csv files and extracts the images from them and returns them
    images = []
    data = pd.read_csv(path)

    for i in range(0, data.shape[0]):
        row = data.iloc[i].to_numpy()
        grid = np.reshape(row, (28, 28))
        images.append(getBlWhImage(grid))
    return images

def getData(num1, num2):
    location = "train_and_valid/"
    trainFile1 = "train" + num1
    trainFile2 = "train" + num2
    validFile1 = "valid" + num1
    validFile2 = "valid" + num2
    trainFiles = []
    validFiles = []
    for file in os.listdir(location):#get all file names
        try:
            if file.startswith(trainFile1) or file.startswith(trainFile2) and file.endswith(".csv"): 
                trainFiles.append(file)

            elif file.startswith(validFile1) or file.startswith(validFile2) and file.endswith(".csv"): 
                validFiles.append(str(file))

        except Exception as e:
            raise e
        trainP = []
        trainL = []
        testP = []
        testL = []
    print("Setting up decision tree, please wait")
    for i in range(len(trainFiles)):
        temp0 = []
        temp1 = []
        temp2 = []
        temp3 = []

        temp0,temp1 = open_images(os.path.join(location, trainFiles[i]))
        temp2,temp3 = open_images(os.path.join(location, validFiles[i]))

        trainP.append(temp1)
        trainL.append(temp0)
        testP.append(temp3)
        testL.append(temp2)
    #test = open_test("test.csv")
    return trainP, trainL, testP, testL, []

def testData(Test, w, num1, num2):
    correct = 0
    for value in Test:
            digits = []
            label = int(value[11])
            bias = value[10]
            #value [:10] grabs the 10 columns with feature data
            #np.dot takes the dot product of the features vector and the given weight
            digits.append(np.sum(np.dot(w[num1], value[:10])))
            digits.append(np.sum(np.dot(w[num2], value[:10])))
            
            if digits[0] > digits[1]:
                guess = num1
            else:
                guess = num2

            if guess == label:
                correct += 1
    print("VALIDATION DATA - num correct over total: ", correct, "/", len(Test), "accuracy: ", correct/len(Test))

def guess(Test, w):
    guesses = []
    for value in Test:
            digits = []
            #value [:10] grabs the 10 columns with feature data
            #np.dot takes the dot product of the features vector and the given weight
            digits.append(np.sum(np.dot(w[0], value[:10])))
            digits.append(np.sum(np.dot(w[1], value[:10])))
            digits.append(np.sum(np.dot(w[2], value[:10])))
            digits.append(np.sum(np.dot(w[3], value[:10])))
            digits.append(np.sum(np.dot(w[4], value[:10])))
            digits.append(np.sum(np.dot(w[5], value[:10])))
            digits.append(np.sum(np.dot(w[6], value[:10])))
            digits.append(np.sum(np.dot(w[7], value[:10])))
            digits.append(np.sum(np.dot(w[8], value[:10])))
            digits.append(np.sum(np.dot(w[9], value[:10])))
            guess = digits.index(max(digits))
            guesses.append(guess)
    return guesses

def train(Train, w, epochs):
    best_fit_val = 0
    best_fit = []
    best_epoch = 0
    for i in range (1,epochs+1): #will use epochs here later
        correct = 0
        for value in Train:
            digits = []
            label = int(value[11])
            bias = value[10]
            #value [:10] grabs the 10 columns with feature data
            #np.dot takes the dot product of the features vector and the given weight
            digits.append(np.sum(np.dot(w[0], value[:10])))
            digits.append(np.sum(np.dot(w[1], value[:10])))
            digits.append(np.sum(np.dot(w[2], value[:10])))
            digits.append(np.sum(np.dot(w[3], value[:10])))
            digits.append(np.sum(np.dot(w[4], value[:10])))
            digits.append(np.sum(np.dot(w[5], value[:10])))
            digits.append(np.sum(np.dot(w[6], value[:10])))
            digits.append(np.sum(np.dot(w[7], value[:10])))
            digits.append(np.sum(np.dot(w[8], value[:10])))
            digits.append(np.sum(np.dot(w[9], value[:10])))
            guess = digits.index(max(digits))
            if guess != label:
                for k in range(10):
                    temp = w[label][k]
                    w[label][k] += w[guess][k]*(0.05 / (1.25*i))
                    w[guess][k] -= temp*(0.05 / (1.25*i))
            else:
                correct += 1
        if best_fit_val < correct:
            best_fit_val = correct
            best_fit = w
            best_epoch = i
        print("correct over total", correct, "/", len(Train), " source epoch:", i)
    print("best weights found:")
    for weight in best_fit:
        print(weight)
    print("Accuracy with given weights", best_fit_val/len(Train), "source epoch: ", best_epoch)
    return best_fit

def main(num1, num2):
    trainP, trainL, testP, testL, test = getData(num1, num2)
    TrainP, TrainL = setupData(trainP, trainL)#sets up TRAIN
    TestP, TestL = setupData(testP, testL)#sets up TEST
    TrainP = np.array(TrainP)
    TrainL = np.array(TrainL)
    TestP = np.array(TestP)
    TestL = np.array(TestL)
    weights = []
    for i in range(10):#set up weights
        temp = []
        for j in range(10):
            temp.append(round(random.uniform(-0.05, 0.05),8))
        weights.append(temp)
    #bestWeights = train(trainingData, weights, epochs)
    #testData(testingData, bestWeights, int(num1), int(num2))
    #print("Guesses for test.csv")
    #print(guess(test, bestWeights))
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(TrainP, TrainL)

    y_pred = clf.predict(TestP)
    TreeAccuracy = accuracy_score(TestL, y_pred)
    print("Tree model accuracy between", num1, "and", num2, ":", TreeAccuracy, "\n")

if __name__ == "__main__":
    main(1, 2)