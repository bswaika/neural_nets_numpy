from Loss import Loss
import numpy as np
import matplotlib.pyplot as plt
from progress import progress
from Layer import DenseLayer

class Network:
    def __init__(self, alpha, layers, loss, regularization=False, lambd=0):
        self.alpha = alpha
        self.layers = layers
        self.num_layers = len(self.layers)
        self.loss = Loss.l()[loss]
        self.d_loss = Loss.d_l()[loss]
        self.training_losses = []
        self.regularization = regularization
        self.lambd = lambd

    def select(self, X, Y, k, batch_size=0):
        return (X[:, k:k+batch_size], Y[:, k:k+batch_size]) if batch_size else (X[:, k:], Y[:, k:])

    def train(self, X, Y, epochs, batch_size=0):
        print('Training started...')
        for j in range(epochs):
            if not batch_size:
                batch_size = len(X[0])
            
            l = len(X[0])
            k = 0
            while k < l:
                if l - k >= batch_size:
                    FEATURES, LABEL = self.select(X, Y, k, batch_size)
                    # print(FEATURES.shape, LABEL.shape)
                    m = batch_size
                else:
                    FEATURES, LABEL = self.select(X, Y, k, 0)
                    # print(FEATURES.shape, LABEL.shape)
                    m = l - k
                
                i = 0
                A = FEATURES
                while i < self.num_layers:
                    A = self.layers[i].forward(A)
                    if type(self.layers[i]) == DenseLayer:
                        if self.layers[i].drop:
                            A = self.layers[i].dropout()
                    i += 1
                
                L = self.loss(A, LABEL)
                J = np.sum(L) / m
                if self.regularization:
                    regularizer = sum([layer.compute_frobenius_norm() for layer in self.layers])
                    J += ((self.lambd * regularizer) / (2 * m))

                self.training_losses.append(J)

                i = self.num_layers - 1
                dA = self.d_loss(A, LABEL)
                while i >= 0:
                    A = self.layers[i-1].A if i > 0 else FEATURES
                    dA = self.layers[i].backward(A, dA, self.lambd)
                    i -= 1                
                
                for layer in self.layers:
                    layer.update(self.alpha, m)

                k += batch_size
            
            progress(j, epochs)
        print()
        print('Training completed...')

    def predict(self, X):
        i = 0
        A = X
        while i < self.num_layers:
            A = self.layers[i].forward(A)
            i += 1
        return A

    def measure(self, A, Y):
        preds = np.argmax(A, axis=0)
        labels = np.argmax(Y, axis=0)
        return np.sum(preds == labels) / len(preds)

    def create_conf_matrix(self, A, Y):
        conf_matrix = [[0 for _ in range(A.shape[0])] for _ in range(A.shape[0])]
        preds = np.argmax(A, axis=0)
        labels = np.argmax(Y, axis=0)
        for p, l in zip(preds, labels):
            conf_matrix[l][p] += 1
        self.conf_matrix = conf_matrix

    def test(self, X, Y):
        A = self.predict(X)
        acc = self.measure(A, Y)
        self.create_conf_matrix(A, Y)
        return acc

    def plot_conf_matrix(self, ax):
        # x = np.arange(-0.5, 10, 1)
        # y = np.arange(-0.5, 10, 1)
        ax.set_title('Confusion Matrix')
        ax.imshow(self.conf_matrix)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, self.conf_matrix[i][j], ha="center", va="center", color="w")

    def plot_training_loss(self, ax):
        ax.set_title('Training Loss')
        ax.plot(np.arange(len(self.training_losses)), self.training_losses, 'r', linewidth=0.5)
        ax.set(ylabel='Loss', xlabel='Epochs')

    def plot_accuracy(self, ax, acc, title):
        ax.set_title(title)
        ax.bar(np.arange(0, len(acc)), [a * 100 for a in acc])
        for i, a in enumerate(acc):
            ax.text(i, a+0.5, str(round(a * 100, 3)), ha='center')
        ax.set_yticks(np.arange(0,105,5))
        ax.set(ylabel='Percentage', xlabel='Iteration')

    def plot_model_stats(self, filename, train_acc, test_acc, title, subtitle):
        plt.close()
        # fig = plt.figure()
        plt.rcParams["figure.figsize"] = (25,25)
        plt.rcParams["font.size"] = 20
        plt.suptitle(title)
        plt.figtext(0.3, 0.95, subtitle)
        ax1 = plt.subplot(221)
        self.plot_training_loss(ax1)
        ax2 = plt.subplot(222)
        self.plot_conf_matrix(ax2)
        ax3 = plt.subplot(223)
        self.plot_accuracy(ax3, train_acc, 'Training Accuracy')
        ax4 = plt.subplot(224)
        self.plot_accuracy(ax4, test_acc, 'Testing Accuracy')
        plt.savefig(f'../{filename}')

        