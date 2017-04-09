
import numpy as np
import matplotlib.pyplot as plt
from util import getMNISTData
from sklearn.decomposition import PCA


def main():
    # Load the Data
    X_train, y_train, X_test, y_test = getMNISTData()

    # Create an objec of type PCA
    pca = PCA()

    # Fit the model to X_Train and do Dimensionality Reduction on X_train
    reduced_dimension = pca.fit_transform(X_train)
    # Plot fitted data
    # [:,0]: x-dimension; [:,1]: y-dimension; c: color
    plt.scatter(reduced_dimension[:,0], reduced_dimension[:,1], s=100, c=y_train, alpha=0.5)
    plt.title('PCA Dimensionality Reduction')
    plt.show()

    # Plot the percentage of variance explained by each selected component
    plt.plot(pca.explained_variance_ratio_)
    plt.title('Explained Variance Ratio')
    plt.show()

    # 2. Cumilative Variance
    # k: number of dimensions that give us 95-99% variance
    # Plot all Eigen Values Cumulatively
    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]
    plt.plot(cumulative)
    plt.title('Cumulative Variance')
    plt.show()


# ----------------- Testing ---------------------
if __name__ == '__main__':
    main()

