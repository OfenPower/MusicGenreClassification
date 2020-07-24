from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.cm as cm


if __name__ == "__main__":

    # Digit Dataset laden
    digits = datasets.load_digits()
    
    # Take the first 500 data points: it's hard to see 1500 points
    X = digits.data[:500]
    y = digits.target[:500]

    # Fit and transform with a TSNE
    tsne = TSNE(n_components=2, random_state=0)

    # Project the data in 2D
    X_2d = tsne.fit_transform(X)

    # Visualize the data
    target_ids = range(len(digits.target_names))

    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, digits.target_names):
        print("y: {}, i: {}".format(y, i))
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()