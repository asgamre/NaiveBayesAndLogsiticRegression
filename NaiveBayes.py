import os
from os.path import join
import re
from collections import Counter
import math
import Main


def train(docs,classes,path,stopwords):
    spamCount = Counter()
    hamCount = Counter()
    totalWords = Counter()
    vocab = set()
    labelFeatures = []
    # print docs
    for testFiles in docs:
        label = classes[docs.index(testFiles)]
        root = path[docs.index(testFiles)]
        for testFile1 in testFiles:
            wordCount = Counter()
            for word in open(join(root,testFile1)).read().split():
                if word not in stopwords and re.match("^[a-zA-Z]*$", word):
                    vocab.add(word)
                    wordCount[word]+=1
                    if label == "ham":
                        i=0
                        hamCount[word]+=1
                    else:
                        i=1
                        spamCount[word]+=1
                    totalWords[i]+=1
            labelFeatures.append(Main.Features(label, wordCount))
            # print wordCount[word]
    # print totalWords[0]
    # print totalWords[1]
    wordProb = {}
    vocabLength = len(vocab)

    for word in vocab:
        if word in hamCount:
            ham = float(hamCount[word] + 1) / float(totalWords[0] + vocabLength)
        else:
            ham = float(1) / float(totalWords[0] + vocabLength)
        if word in spamCount:
            spam = float(spamCount[word] + 1) / float(totalWords[0] + vocabLength)
        else:
            spam = float(1) / float(totalWords[1] + vocabLength)
        wordProb[word] = [ham, spam]
    return labelFeatures,vocab,wordProb


def predict(testFile,stopwords,wordProb,prior):
    spam=0.0
    ham=0.0
    for word in open(testFile).read().split():
        if word not in stopwords and re.match("^[a-zA-Z]*$", word) and word in wordProb:
            eachWordStats = wordProb[word]
            ham +=math.log(eachWordStats[0])
            spam += math.log(eachWordStats[1])
    # print "prior in predict"
    # print prior[0]
    ham +=math.log(prior["ham"])
    spam +=math.log(prior["spam"])
    if ham >= spam:
        return "ham"
    else:
        return "spam"



def Accuracy(test,stopwords,wordProb,prior):
    print "Enter Accuracy"
    correctcount = 0
    totaltestFiles = 0
    # print test
    for root, dirs, testFiles in os.walk(test):
        print "root in NB"
        print root
        if testFiles:
            # print "testFiles"
            # print testFiles
            for testFile in testFiles:
                totaltestFiles+=1
                # actuallabel = root.split('\\')[-1]

                actuallabel = root.split('/')[-1]
                predictedlabel=predict(join(root,testFile),stopwords,wordProb,prior)
                if actuallabel==predictedlabel:
                    correctcount+=1
    return float(correctcount)/float(totaltestFiles)*100

