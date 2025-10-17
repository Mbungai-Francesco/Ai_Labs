from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning, DictionaryLearning, NMF
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold

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

#plot_digits_examples(x_digits, y_digits)


#print(f'Original features shape: {x_digits.shape}')

thr = 14
selector = VarianceThreshold(threshold=thr)

# Apply the threshold on the digits data
x_digits_thresholded = selector.fit_transform(x_digits)

#print(f'Features shape after threshold: {x_digits_thresholded.shape}')

#print(f"{x_digits.shape[1] - x_digits_thresholded.shape[1]} features were removed")

# Get variances from the fitted selector
variances = selector.variances_
# Boolean mask: True for features to keep (variance strictly greater than threshold)
mask = variances > thr
#print(mask)

def plot_digits_examples_thresholded(x_digits, y_digits, mask, n_class=10):
    """
    Plots examples of digits along with their corresponding masks.
    Parameters:
    - x_digits (numpy.ndarray): Array of digit images.
    - y_digits (numpy.ndarray): Array of digit labels.
    - mask (numpy.ndarray): Array representing the mask to be applied to the digits.
    - n (int): Number of classes. Should be the same than in load_digits. Default is 10.
    """
    fig = plt.figure(figsize=(10,5))

    for idx_class in range(n_class): # Parse all classes
        # Index of the current example in the final patchwork
        plt.subplot(3, n_class, idx_class+1)

        # Pick a random digit in the current category     
        curX = x_digits[y_digits==idx_class]
        r = np.random.randint(curX.shape[0])
        curim = curX[r, :].reshape((8,8))

        # Display the digit (and remove ticks)
        plt.imshow(curim, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])

        # Do the same, but for the mask
        plt.subplot(3, 10, idx_class+11)
        plt.imshow(mask.reshape((8,8)), cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])

        # Do the same, but for the masked digit
        plt.subplot(3,10,idx_class+21)
        curim_masked =  curim*mask.reshape((8,8))
        plt.imshow(curim_masked,cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])

    plt.show()
    
plot_digits_examples_thresholded(x_digits, y_digits, mask)

# Compute and plot histogram of feature variances
n_total = variances.size
n_kept = (variances > thr).sum()
print(f"{n_kept}/{n_total} features kept ({n_kept/n_total:.1%}) with threshold={thr}")

plt.figure(figsize=(8,4))
plt.hist(variances, bins=30, color='C0', edgecolor='k', alpha=0.7)
plt.axvline(thr, color='r', linestyle='--', linewidth=2, label=f'threshold = {thr}')
plt.xlabel('Feature variance')
plt.ylabel('Count')
plt.title('Histogram of feature variances')
plt.legend()
plt.grid(alpha=0.3)
#plt.show()

### - Use the numpy percentile function to find the threshold to remove a given percentages of features to keep.
q = np.percentile(variances, 75)
print(f"Threshold to keep 25% of features: {q}")

### - Find what is, empirically, the best threshold in your opinion.

