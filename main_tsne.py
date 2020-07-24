from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import json
import os
import numpy as np

import file_processing


if __name__ == "__main__":

    # Daten aus json laden
    JSON_PATH = "../data_adjusted_all_n10.json"
    inputs, targets = file_processing.load_data(JSON_PATH)
    inputs_2d = inputs.reshape(inputs.shape[:-2] + (-1,))   # 3D Array zu 2D Array umwandeln
    print(inputs_2d.shape)
    #inputs_mfccs = inputs[0, :, :]     # erste Dimension löschen

    # Fit and transform with PCA and TSNE and project the data in 2D
    #pca = PCA(n_components=200) # 1292 # 1690 (50 sind ca. 2% davon)
    #principalComponents = pca.fit_transform(inputs_2d)
    tsne = TSNE(n_components=2, perplexity=100, verbose=1, n_iter=1000)  # random_state=0
    X_2d = tsne.fit_transform(inputs_2d)

    # Visualize the data
    target_ids = range(10)  # Labels gehen von 0-9
    target_ids2 = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    plt.figure(figsize=(6, 5))
    #colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i, c, label in zip(target_ids, colors, target_ids2):
        #print("i: {}, targets: {}, c: {}, label: {}".format(i, targets, c, target_ids2))
        plt.scatter(X_2d[targets == i, 0], X_2d[targets == i, 1], c=c, label=label)
    plt.legend()
    plt.show()

