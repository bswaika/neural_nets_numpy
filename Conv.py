import numpy as np
import Activation

class FlattenLayer:
    def __init__(self, dims):
        self.dims = dims

    def compute_frobenius_norm(self):
        return 0
    
    def forward(self, A):
        self.A = []
        for example in A:
            self.A.append(example.flatten())
        self.A = np.transpose(np.asarray(self.A))
        return self.A
    
    def backward(self, A, dA, lambd=0):
        result = []
        dA = np.transpose(dA)
        for example in dA:
            # print(example)
            result.append(example.reshape(self.dims))
        return np.asarray(result)
    
    def update(self, alpha, m):
        return

class PoolingLayer:
    def __init__(self, inputs, channels, kernel, stride, type='max'):
        self.kernel = kernel
        self.stride = stride
        self.type = type
        self.inputs = inputs
        self.channels = channels

    def compute_frobenius_norm(self):
        return 0
    
    def max_pool(self, A, kernel, s, type='max'):
        result_dims = ((((A.shape[0] - kernel) // s) + 1), (((A.shape[1] - kernel) // s) + 1), A.shape[2])
        result = np.zeros(result_dims)

        args = None
        if type=='max':
            args = [[[0 for _ in range(result_dims[2])] for _ in range(result_dims[1])] for _ in range(result_dims[0])]

        for k in range(A.shape[2]):
            for i, A_i in zip(range(result.shape[0]), range(0, A.shape[0], s)):
                for j, A_j in zip(range(result.shape[1]), range(0, A.shape[1], s)):
                    if type == 'max':
                        # print(A[A_i:A_i+kernel, A_j:A_j+kernel, k])
                        result[i, j, k] = A[A_i:A_i+kernel, A_j:A_j+kernel, k].max()
                        args[i][j][k] = np.unravel_index(np.argmax(A[A_i:A_i+kernel, A_j:A_j+kernel, k]), (kernel, kernel))
                        # args[A_i][A_j][k] = A[A_i:A_i+kernel, A_j:A_j+kernel, k] == result[i, j, k]
                    elif type=='avg':
                        result[i, j, k] = np.sum(A[A_i:A_i+kernel, A_j:A_j+kernel, k]) / (kernel ** 2)
        
        return result, args

    def de_pool(self, A, dA, idx):
        # print(A.shape, dA.shape)
        result = np.zeros(A.shape)
        for k in range(A.shape[2]):
            for i, A_i in zip(range(dA.shape[0]), range(0, A.shape[0], self.stride)):
                for j, A_j in zip(range(dA.shape[1]), range(0, A.shape[1], self.stride)):
                    if A_i + self.kernel < A.shape[0] and A_j + self.kernel < A.shape[1]:
                        if self.type == 'max':
                            result[A_i+self.args[idx, i, j, k, 0], A_j+self.args[idx, i, j, k, 1], k] += dA[i, j, k]
                        elif self.type == 'avg':
                            result[A_i:A_i+self.kernel, A_j:A_j+self.kernel, k] += dA[i, j, k] / (self.kernel ** 2)
        return result
        
    def forward(self, A):
        result = []
        if self.type == 'max':
            self.args = []
            for example in A:
                res, args = self.max_pool(example, self.kernel, self.stride, self.type)
                result.append(res)
                self.args.append(args)
            self.args = np.asarray(self.args)
        elif self.type == 'avg':    
            for example in A:
                res, args = self.max_pool(example, self.kernel, self.stride, self.type)
                result.append(res)
        self.A = np.asarray(result)
        return self.A

    def backward(self, A, dA, lambd=0):
        result = []
        for example in range(A.shape[0]):
            result.append(self.de_pool(A[example], dA[example], example))
        return np.asarray(result)

    def update(self, alpha, m):
        return

class ConvLayer:
    def __init__(self, inputs, channels, kernel, num_filters, activation, stride=1, type='valid', padding=0):
        self.inputs = inputs
        self.channels = channels
        self.kernel = kernel
        self.num_filters = num_filters
        self.stride = stride
        self.type = type
        self.padding = padding
        self.W = np.random.randn(self.num_filters, self.kernel, self.kernel, self.channels)
        self.B = np.zeros((1, self.num_filters))
        self.g = Activation.g()[activation]
        self.d_g = Activation.d_g()[activation]
    
    def convolve(self, A, W, s=1, type='valid', p=0):
        if type == 'same':
            p = ((A.shape[0] * (s-1)) + W.shape[0] - s) // 2
        if type == 'full':
            p = W.shape[0] - 1
            
        A = np.pad(A, ((p, p), (p, p), (0, 0)), 'constant', constant_values=0)
        # print(A)

        result_dims = ((((A.shape[0] - W.shape[0]) // s) + 1), (((A.shape[1] - W.shape[1]) // s) + 1), A.shape[2])
        result = np.zeros(result_dims)

        # print(A.shape, W.shape, result.shape)
        for i, A_i in zip(range(result.shape[0]), range(0, A.shape[0], s)):
            for j, A_j in zip(range(result.shape[1]), range(0, A.shape[1], s)):
                result[i, j, :W.shape[2]] = np.sum(np.sum(A[A_i:A_i+W.shape[0], A_j:A_j+W.shape[1], :W.shape[2]] * W[:W.shape[0], :W.shape[1], :W.shape[2]], axis = 1), axis=0)

        return np.sum(result, axis=2)
    
    def compute_frobenius_norm(self):
        norm = self.W * self.W
        return np.sqrt(np.sum(np.sum(np.sum(np.sum(norm, axis=3), axis=2), axis=1), axis=0))

    def forward(self, A):
        result = []
        for example in A:
            r = []
            for filter in range(self.num_filters):
                r.append(self.convolve(example, self.W[filter], self.stride, self.type, self.padding))
            result.append(r)
        result = np.asarray(result)
        result = np.transpose(result, axes=(0, 2, 3, 1))
        self.Z = result + self.B
        self.A = self.g(self.Z)
        self.A = np.clip(self.A, -500, 500)
        return self.A
    
    def backward(self, A, dA, lambd=0.2):
        self.dZ = dA * self.d_g(self.Z)
        
        unfolded_dA = np.transpose(dA, axes=(0, 3, 1, 2))
        result = []
        for example in range(A.shape[0]):
            r = []
            for filter in range(self.num_filters):
                da = np.copy(unfolded_dA[example][filter])
                da = np.tile(da.reshape(da.shape[0], da.shape[1], 1), A.shape[3])
                r.append(self.convolve(A[example], da, self.stride, self.type, self.padding))
            result.append(r)
        result = np.asarray(result)
        result = np.sum(result, axis=0) / A.shape[0]
        self.dW = np.tile(result.reshape(result.shape[0], result.shape[1], result.shape[2], 1), A.shape[3]) + (lambd * self.W)
        self.dB = np.sum(np.sum(np.sum(self.dZ, axis=2), axis=1), axis=0, keepdims=True)
        
        
        # print(self.dW.shape)

        result = []
        for example in range(A.shape[0]):
            r = []
            for filter in range(self.num_filters):
                r.append(self.convolve(dA[example], np.rot90(self.W[filter], 2), self.stride, 'full'))
            result.append(r)
        result = np.asarray(result)
        result = np.sum(np.transpose(result, axes=(0, 2, 3, 1)), axis=3, keepdims=True) / self.num_filters
        result = np.tile(result.reshape(result.shape[0], result.shape[1], result.shape[2], 1), A.shape[3])
        
        return result

    def update(self, alpha, m):
        self.W -= (alpha * (self.dW / m))
        self.B -= (alpha * (self.dB / m))