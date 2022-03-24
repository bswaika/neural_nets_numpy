import numpy as np
import warnings

warnings.filterwarnings('error')

class Loss:
    @staticmethod
    def l():
        return {
            'log_likelihood': Loss.likelihood,
            'mse': Loss.mse,
            'cross_entropy': Loss.cross_entropy 
        }
    
    @staticmethod
    def d_l():
        return {
            'log_likelihood': Loss.d_likelihood,
            'mse': Loss.d_mse,
            'cross_entropy': Loss.d_cross_entropy
        }

    @staticmethod
    def likelihood(A, Y):
        return -((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))
    
    @staticmethod
    def d_likelihood(A, Y):
        return -(Y / A) + ((1 - Y) / (1 - A))

    @staticmethod
    def mse(A, Y):
        return ((Y - A) ** 2) / 2
    
    @staticmethod
    def d_mse(A, Y):
        return -2 * A * (Y - A)

    @staticmethod
    def cross_entropy(A, Y):
        clip_low = 0.00000000000000000000000000000000000000000001
        A = np.clip(A, clip_low, np.max(A))
        return -np.sum(Y * np.log(A), axis=0)
    
    @staticmethod
    def d_cross_entropy(A, Y):
        clip_low = 0.00000000000000000000000000000000000000000001
        A = np.clip(A, clip_low, np.max(A))
        return - Y / A