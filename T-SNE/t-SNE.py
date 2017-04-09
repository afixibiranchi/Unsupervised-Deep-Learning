import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Generate Random Data
def get_data():
    # Number of data points
    N = 600
    R_inner = 10
    R_outer = 20

    R1 = np.random.randn(N/2) + float(R_inner)
    # R1.astype(int)
    theta = 2*np.pi*np.random.random(N/2)
    # theta.astype(int)
    X_inner = np.concatenate([[R1*np.cos(theta)], [R1*np.sin(theta)]])
    # X_inner.astype(int)

    R2 = np.random.randn(N/2) + float(R_outer)
    # R2.astype(int)
    theta = 2*np.pi * np.random.random(N/2)
    # theta.astype(int)
    X_outer = np.concatenate([[R2*np.cos(theta)], [R1*np.sin(theta)]])
    # X_outer.astype(int)

    X = np.concatenate([X_inner, X_outer])
    # X.astype(int)
    y = np.array([0]*(N//2) + [1]*(N//2))
    # y.astype(int)
    return X,y


# Apply TSNE to data
def main():
    X,y = get_data()
    y_c = np.array([0]*int((600/2)) + [1]*int((600/2)))
    plt.scatter(X[:,0], X[:,1], s=100, alpha=0.5)

    tsne = TSNE(perplexity=40)
    Z = tsne.fit_transform(X)
    plt.scatter(Z[:,0],Z[:,1], s=100,alpha=0.5)
    plt.show()



if __name__ == '__main__':
    main()
