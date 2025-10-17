from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning, DictionaryLearning, NMF
from sklearn.manifold import TSNE

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

x_digits, y_digits = load_digits(n_class=10, return_X_y=True)

def plot_digits_examples(x_digits, y_digits, n_class=10, n_examples=3):
    """
    Plot some examples of digits from the given dataset.
    Parameters:
    - x_digits (numpy.ndarray): Array of digit images.
    - y_digits (numpy.ndarray): Array of digit labels. Used for selecting examples.
    - n (int): Number of classes. Should be the same than in load_digits. Default is 10.
    """
    fig = plt.figure(figsize=(10,5))

    for idx_class in range(n_class): # Parse all classes
        for idx_example in range(1,n_examples+1): # Select n_examples examples
            # Index of the current example in the final patchwork
            plt.subplot(n_examples, n_class, n_examples*idx_class+idx_example)

            # pick a random digit in the current category
            curX = x_digits[y_digits==idx_class] 
            r = np.random.randint(curX.shape[0])

            # Reshape the image: controling its size.
            curim = curX[r, :].reshape((8,8))

            # Display the digit (and remove ticks)
            plt.imshow(curim, cmap=plt.cm.gray)
            plt.xticks([])
            plt.yticks([])

    plt.show()

plot_digits_examples(x_digits, y_digits)