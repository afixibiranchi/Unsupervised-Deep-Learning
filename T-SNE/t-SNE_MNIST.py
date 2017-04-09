import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from util import getMNISTData


def main():
    X_train, y_train, x_test, y_test = getMNISTData()

    sample_size = 1000
    X = X_train[:sample_size]
    y = y_train[:sample_size]

    tsne = TSNE()
    Z = tsne.fit_transform(X)
    plt.scatter(Z[:, 0], Z[:, 1], s=100, c=y, alpha=0.5)
    plt.title('TSNE on MNIST')
    plt.show()



if __name__ == '__main__':
    main()