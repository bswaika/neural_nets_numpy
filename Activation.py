import numpy as np

class Activation:
    @staticmethod
    def g():
        return {
            'sigmoid': Activation.sigmoid,
            'relu': Activation.relu,
            'softmax': Activation.softmax 
        }

    @staticmethod
    def d_g():
        return {
            'sigmoid': Activation.d_sigmoid,
            'relu': Activation.d_relu,
            'softmax': Activation.d_softmax
        }

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def d_sigmoid(Z):
        return (Activation.sigmoid(Z) * (1 - Activation.sigmoid(Z)))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def d_relu(Z):
        return (Z > 0)

    @staticmethod
    def softmax(Z):
        shift = np.clip(Z, -500, 500)
        t = np.exp(shift)
        return t / np.sum(t, axis=0, keepdims=True)

    @staticmethod
    def d_softmax(Z):
        Z_T = np.transpose(Z)
        result = []
        for z in Z_T:
            SM = z.reshape(-1, 1) 
            J = np.diagflat(z) - np.dot(SM, SM.T)
            result.append(J)
        return np.asarray(result)