import math
import os
from os.path import join
from collections import Counter
import re


def trainLR(features, vocab):

    for feature in features:
        feature.addFeatures(1)
        for word in vocab:
            if word in feature.wordCount:
                feature.addFeatures(feature.wordCount[word])
            else:
                feature.addFeatures(0)

    weightVector = []
    for i in range(len(vocab) + 1):
        weightVector.append(0)

    eeta = 0.005
    lamb = 0.5

    for i in range(1, 150):
        error = []
        for feature in features:
            if feature.label == 'ham':
                actualY = 1
            else:
                actualY = 0
            wixi = 0.0
            for x in range(len(vocab) + 1):
                wixi += weightVector[x] * feature.features[x]
            predY = math.exp(wixi)/ (1 + math.exp(wixi))
            error.append(actualY-predY)
        for index, w in enumerate(weightVector):
            allError = 0.0
            for insIndex, feature in enumerate(features):
                allError += feature.features[index] * error[insIndex]
            weightVector[index] = (weightVector[index] + (eeta * allError) - (eeta * lamb * weightVector[index]))
    return weightVector

def predictLR(weights, featureVector):

    wixi = 0.0
    for index, w in enumerate(weights):
        wixi += (weights[index] * featureVector[index])

    hamCount = float(math.exp(wixi)) / float(1 + math.exp(wixi))
    spamCount = 1 - hamCount
    if hamCount >= spamCount:
        return 'ham'
    else:
        return 'spam'

def AccuracyLR(testFile, weights, vocab, stopWords):

    numOfFiles , accuracy = 0, 0
    for root, dirs, files in os.walk(testFile):
        # print "root in LR"
        # print root
        if files:
            for file in files:
                numOfFiles += 1
                # actualLabel = root.split('\\')[-1]
                actualLabel = root.split('/')[-1]
                wordCount = Counter()
                for word in open(join(root,file)).read().split():
                    if word not in stopWords and re.match("^[a-zA-Z]*$", word):
                        wordCount[word] += 1
                featureVector = [1]
                for word in vocab:
                    if word in wordCount:
                        featureVector.append(wordCount[word])
                    else:
                        featureVector.append(0)
                predlabel = predictLR(weights, featureVector)
                if predlabel == actualLabel:
                    accuracy += 1
    return float(accuracy)/float(numOfFiles) * 100