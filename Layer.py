import numpy as np
from Activation import Activation

class DenseLayer:
    def __init__(self, inputs, outputs, activation, dropout=False, keep_prob=1):
        self.W = np.random.randn(outputs, inputs)
        self.B = np.zeros((outputs, 1))
        self.g = Activation.g()[activation]
        self.d_g = Activation.d_g()[activation]
        self.drop = dropout
        self.keep_prob = keep_prob

    def compute_frobenius_norm(self):
        norm = self.W * self.W
        return np.sqrt(np.sum(np.sum(norm, axis=1, keepdims=True), axis=0))

    def forward(self, A):
        self.Z = np.dot(self.W, A) + self.B
        self.A = self.g(self.Z)
        self.A = np.clip(self.A, -500, 500)
        return self.A

    def dropout(self):
        if self.drop:
            D = np.random.randn(self.A.shape[0], self.A.shape[1]) < self.keep_prob
            # print(self.A)
            self.A *= D
            self.A /= self.keep_prob
        return self.A

    def backward(self, A, dA, lambd=0):
        self.dZ = dA * self.d_g(self.Z)
        self.dW = np.dot(self.dZ, np.transpose(A)) + (lambd * self.W)
        self.dB = np.sum(self.dZ, axis=1, keepdims=True)
        return np.dot(np.transpose(self.W), self.dZ)       

    def update(self, alpha, m):
        self.W -= (alpha * (self.dW / m))
        self.B -= (alpha * (self.dB / m))

class SoftmaxLayer:
    def __init__(self, inputs, outputs):
        self.W = np.random.randn(outputs, inputs)
        self.B = np.zeros((outputs, 1))
        self.g = Activation.g()['softmax']
        self.d_g = Activation.d_g()['softmax']

    def compute_frobenius_norm(self):
        norm = self.W * self.W
        return np.sqrt(np.sum(np.sum(norm, axis=1, keepdims=True), axis=0))

    def forward(self, A):
        self.Z = np.dot(self.W, A) + self.B
        self.A = self.g(self.Z)
        self.A = np.clip(self.A, -500, 500)
        return self.A

    def backward(self, A, dA, lambd=0):
        self.dZ = []
        self.dA = self.d_g(self.A)
        for i in range(self.dA.shape[0]):
            self.dZ.append(np.dot(self.dA[i], dA[:, i]))

        self.dZ = np.asarray(self.dZ).T
        self.dW = np.dot(self.dZ, np.transpose(A)) + (lambd * self.W)
        self.dB = np.sum(self.dZ, axis=1, keepdims=True)
        return np.dot(np.transpose(self.W), self.dZ)       

    def update(self, alpha, m):
        self.W -= (alpha * (self.dW / m))
        self.B -= (alpha * (self.dB / m))