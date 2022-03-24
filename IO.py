import numpy as np
import csv

def parseFeatures():
    X = []
    with open('./data/images.csv') as features:
        reader = csv.reader(features)
        for row in reader:
            X.append(np.asarray(list(map(lambda x: x / 255, map(int, row)))))
    return np.asarray(X).T

def parseLabels():
    Y = []
    with open('./data/labels.csv') as labels:
        reader = csv.reader(labels)
        for row in reader:
            Y.append(list(map(lambda x: [1 if l == x else 0 for l in range(10)], map(int, row)))[0])
    return np.transpose(np.asarray(Y))


def split(X, Y, ratio=0.8):
    t = X.shape[1]
    s = int(ratio * t)
    return ((X[:, :s], Y[:, :s]), (X[:, s:], Y[:, s:]))


def r_select(X, Y):
    assert len(X[1]) == len(Y[1])
    randomize = np.arange(len(X[1]))
    np.random.shuffle(randomize)
    X, Y = X[:, randomize], Y[:, randomize]
    return X, Y

def initialize_data(X, Y):
    X, Y = r_select(X, Y)
    train, test = split(X, Y, 0.14)
    return train, test

def writeOutput(preds, filename):
    with open(filename, 'w') as outfile:
        for pred in preds:
            outfile.write(str(pred) + '\n')