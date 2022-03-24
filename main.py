from IO import *
from Network import Network
from Layer import DenseLayer, SoftmaxLayer
from time import perf_counter

from copy import deepcopy

X = parseFeatures()
Y = parseLabels()

def test_model(model, epochs, batch, iters, subtitle, file):
    train, test = initialize_data(X, Y)
    trainX, trainY = train
    testX, testY = test

    start = perf_counter()
    model.train(trainX, trainY, epochs, batch_size=batch)
    end = perf_counter()
    model.test(testX, testY)

    acc = [[0 for _ in range(iters)] for _ in range(2)]

    for i in range(iters):
        t_model = deepcopy(model)
        train, test = initialize_data(X, Y)
        trainX, trainY = train
        testX, testY = test

        t_model.train(trainX, trainY, epochs, batch_size=batch)
        acc[0][i] = t_model.test(trainX, trainY)
        acc[1][i] = t_model.test(testX, testY)

        del t_model
    
    model.plot_model_stats(f'hw3_iter2/stats/{file}.pdf', acc[0], acc[1], f'Model {file} | {model.alpha} learn_rate | {epochs} epochs | {batch} batch_size | {round(end - start, 3)}s', subtitle)

# 1 - 0.85 accuracy - 20 epochs - 100 batch
# model = Network(0.1, [
#         DenseLayer(784, 30, 'sigmoid'),
#         DenseLayer(30, 20, 'sigmoid'),
#         SoftmaxLayer(20, 10)
#     ], 'cross_entropy', regularization=True, lambd=0.5)

model = Network(0.001, [
        DenseLayer(784, 30, 'sigmoid'),
        DenseLayer(30, 15, 'sigmoid'),
        DenseLayer(15, 10, 'sigmoid'),
        SoftmaxLayer(10, 10)
    ], 'cross_entropy')

test_model(model, 2500, 100, 5, 'Input 784 - Dense 30 - Dense 20 - Dense 10 - Softmax 10', '05')