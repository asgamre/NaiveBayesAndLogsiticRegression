import os
from os.path import join
import re
from collections import Counter
import math
import sys
import NaiveBayes
import LogReg

class Features:

    def __init__(self, label, data):
        self.label = label
        self.wordCount = data
        self.features = []

    def addFeatures(self, data):
        self.features.append(data)

def main():
    # traindirectory = "hw2_train\\"
    # testdirectory = "hw2_test\\"
    traindirectory = "hw2_train/"
    testdirectory = "hw2_test/"
    train = traindirectory + sys.argv[1]
    test = testdirectory + sys.argv[2]
    stopWordsTxt = "stopwords.txt"

    docs = []
    path=[]
    total = 0
    prior = {}

    for root, dirs, files in os.walk(train):
        # print root
        # print root, dirs, files
        if dirs:
            # print "dirs"
            # print dirs
            classes = dirs
        elif files:
            # print "files"
            # print files
            docs.append(files)
            path.append(root)
            total = (total + len(files))
    # print classes
    for c in classes:
        # print classes.index(c)
        prior[c] = float(len(docs[classes.index(c)]))/total
    # print "prior"
    # print prior["spam"]
    stopwords = ['subject', 're:', 'from' , 'to' , 'cc', 'ect', 'the']
    # print stopwords

    labelFeatures,vocab,wordProb = NaiveBayes.train(docs,classes,path,stopwords)
    print "Training Complete"
    accuracyNB = NaiveBayes.Accuracy(test, stopwords, wordProb, prior)
    print "Accuracy of Naive Bayes with Stop Words: ", accuracyNB

    weightVector = LogReg.trainLR(labelFeatures,vocab)
    accuracyLR = LogReg.AccuracyLR(test, weightVector, vocab, stopwords )
    print "Accuracy of Logistic Regression with Stop Words: ", accuracyLR

    for line in open(stopWordsTxt):
        l = line.rstrip('\n')
        stopwords.append(l)
    # print stopwords

    labelFeatures2,vocab,wordProb= NaiveBayes.train(docs, classes, path, stopwords)
    accuracyNB2 = NaiveBayes.Accuracy(test, stopwords, wordProb, prior)
    print "Accuracy of Naive Bayes without Stop Words: ", accuracyNB2

    weightVector = LogReg.trainLR(labelFeatures2, vocab)
    accuracyLR2 = LogReg.AccuracyLR(test, weightVector, vocab, stopwords )
    print "Accuracy of Logistic Regression without Stop Words: ", accuracyLR2

if __name__=="__main__":
    main()
