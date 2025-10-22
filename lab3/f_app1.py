from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning, DictionaryLearning, NMF
from sklearn.manifold import TSNE

import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline    
np.random.seed(0)

## ? Digits dataset

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
    
# plot_digits_examples(x_digits, y_digits)  ## ! First Plot

## ? Unsupervised Feature Selection
    
print(f'Original features shape: {x_digits.shape}')

from sklearn.feature_selection import VarianceThreshold

thr = 0.05
# TODO: you should vary this threshold, and see what happens!
selector = VarianceThreshold(threshold=thr)

# Apply the threshold on the digits data
x_digits_thresholded = selector.fit_transform(x_digits)

print(f'Features shape after threshold: {x_digits_thresholded.shape}')

print(f"{x_digits.shape[1] - x_digits_thresholded.shape[1]} features were removed")

## ? Visualizing the thresholding effect
### - Compute a boolean mask based on the previous VarianceThreshold selector.

### Hint: you should use the variances computed by the selector.
### You can use the following attribute:
variances = selector.variances_

mask = variances > thr

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
# plot_digits_examples_thresholded(x_digits, y_digits, mask, 10) ## ! Second Plot

## ? Selecting the threshold 
### - Compute a histogram of the feature variances.

## ! Plotting histogram of variances
# plt.figure(figsize=(10, 6))
# plt.hist(variances, bins=50, edgecolor='black')
# plt.xlabel('Variance')
# plt.ylabel('Frequency')
# plt.title('Distribution of Feature Variances')
# plt.axvline(thr, color='red', linestyle='--', label=f'Current threshold = {thr}')
# plt.legend()
# plt.grid(True, alpha=0.3)

print(f"Variance statistics:")
print(f"  Min: {variances.min():.2f}")
print(f"  Max: {variances.max():.2f}")
print(f"  Mean: {variances.mean():.2f}")
print(f"  Median: {np.median(variances):.2f}")
# plt.show()  ## ! Third Plot

### - Use the numpy percentile function to find the threshold to remove a given percentage of features to keep.

# To REMOVE 75% of features, we need to keep the top 25%
# This means we want the 75th percentile as our threshold
percentages_to_remove = [50, 75, 80, 90, 95]

print("Threshold values for different percentages of features to remove:\n")
for pct in percentages_to_remove:
    threshold = np.percentile(variances, pct)
    features_kept = np.sum(variances > threshold)
    features_removed = x_digits.shape[1] - features_kept
    print(f"Remove {pct}% of features:")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Features kept: {features_kept}/{x_digits.shape[1]}")
    print(f"  Features removed: {features_removed}/{x_digits.shape[1]}")
    print()
    
### - Find what is, empirically, the best threshold in your opinion.

# Let's test different percentiles and visualize the results
test_percentiles = [50, 75, 85, 90, 95]

fig = plt.figure(figsize=(15, len(test_percentiles) * 3))

for idx, pct in enumerate(test_percentiles):
    threshold = np.percentile(variances, pct)
    mask_test = variances > threshold
    
    # Show 3 example digits with this threshold
    for digit_class in [0, 3, 8]:  # Show a few representative digits
        plt.subplot(len(test_percentiles), 3, idx * 3 + [0, 3, 8].index(digit_class) + 1)
        
        # Pick a random digit
        curX = x_digits[y_digits == digit_class]
        r = np.random.randint(curX.shape[0])
        curim = curX[r, :].reshape((8, 8))
        
        # Apply mask
        curim_masked = curim * mask_test.reshape((8, 8))
        
        plt.imshow(curim_masked, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        
        if digit_class == 0:
            features_kept = np.sum(mask_test)
            plt.ylabel(f'{pct}% removed\n({features_kept} features)', fontsize=10)
        
        if idx == 0:
            plt.title(f'Digit {digit_class}')

# plt.tight_layout() ## ! Fourth Plot
# plt.show() ## ! Fifth Plot

# My empirical choice (you should adjust based on your observations)
best_threshold_percentile = 85  # This removes 85% of features
best_threshold = np.percentile(variances, best_threshold_percentile)

print(f"\nMy empirical choice:")
print(f"  Remove {best_threshold_percentile}% of features")
print(f"  Threshold: {best_threshold:.2f}")
print(f"  Features kept: {np.sum(variances > best_threshold)}/{x_digits.shape[1]}")

# Visualize with the chosen threshold
selector_best = VarianceThreshold(threshold=best_threshold)
x_digits_best = selector_best.fit_transform(x_digits)
mask_best = selector_best.variances_ > best_threshold

# plot_digits_examples_thresholded(x_digits, y_digits, mask_best, 10) ## ! Sixth Plot


## ? K means clustering
### - Estimate the Kmeans clustering.

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(x_digits)

print(f"K-means clustering completed with {n_clusters} clusters")

### ! Find the cluster centroids.

# Get the cluster centroids
centroids = kmeans.cluster_centers_

print(f"Centroids shape: {centroids.shape}")
print(f"Each centroid has {centroids.shape[1]} features (one per pixel)")

# Visualize the centroids
fig = plt.figure(figsize=(12, 5))

for i in range(n_clusters):
    plt.subplot(2, 5, i + 1)
    
    # Reshape centroid to 8x8 image
    centroid_image = centroids[i].reshape(8, 8)
    
    plt.imshow(centroid_image, cmap=plt.cm.gray)
    plt.title(f'Cluster {i}')
    plt.xticks([])
    plt.yticks([])

# ! Cluster plot
# plt.suptitle('K-means Cluster Centroids', fontsize=14)
# plt.tight_layout()
# plt.show()

# ? Select 10 random examples (one from each class)
whichex = np.random.randint(low=0, high=100, size=1) 
X_samp = np.concatenate([x_digits[y_digits==i][whichex] for i in range(10)])

print(f"X_samp shape: {X_samp.shape}")
print(" ")

### TODO:
### - Use the transform method from the kmeans object on X_samp.
### - Use the argmin method from numpy to generate array with index of closest centroid
### - Fetch the corresponding centroid in another array closest_centroids.
### - Calculate the distances of each sample to its closest centroid using np.min

# (10,10) array containing the distances of each sample to each centroid
distances = kmeans.transform(X_samp)

# (10,) array containing the index of closest centroid to the samples 
idx_closest_centroids = np.argmin(distances, axis=1)

# (10,64) array containing the closest centroid to each sample 
closest_centroids = centroids[idx_closest_centroids]

# (10,) array containing the distance of each sample to its closest centroid
smallest_distances = np.min(distances, axis=1)

print(f'Distances array shape: {distances.shape}, should be (10,10).')
print(f'Indexes of closest centroids array shape {idx_closest_centroids.shape}, should be (10,).')
print(f'Closest centroids array shape {closest_centroids.shape}, should be (10,64).')
print(f'Smallest distances array shape {smallest_distances.shape}, should be (10,).')

### ? Check that the indices of your closest centroids are the same than the ones obtained using the predict method.

print(f'Closest centroids according to your code\t: {idx_closest_centroids}')
print(f'Closest centroids according to sklearn\t\t: {kmeans.predict(X_samp)}')

# ? Plot the samples and their closest centroids

def plot_digits_and_closest_centroids(X_samp, closest_centroids, smallest_distances):
    """
    Plot the original digits and their closest centroids.
    Parameters:
    - X_samp (array-like): Array of original digits.
    - closest_centroids (array-like): Array of closest centroids for each digit.
    - smallest_distances (array-like): Array of smallest distances between each digit and its closest centroid.
    """
    plt.figure(figsize=(20,10))
    for i,(im,im_cen,distance) in enumerate(zip(X_samp, closest_centroids, smallest_distances)):

        plt.subplot(4, 6, 1+2*i)
        plt.imshow(im.reshape(8, 8), cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.title("Original")

        plt.subplot(4, 6, 2+2*i)
        plt.imshow(im_cen.reshape(8, 8), cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.title("Closest centroid, distance %.2E"%distance)
    plt.show()

# plot_digits_and_closest_centroids(X_samp, closest_centroids, smallest_distances) ## ! Seventh Plot

### TODO: Elbow method
### - Generate KMeans models with varying n_clusters
### - Fit each model to the data 
### - Add its inertia to a dedicated list
### - Finally, plot it.

inertias = []
n_clusters_range = range(1, 100)

for n in n_clusters_range:
    kmeans_test = KMeans(n_clusters=n, random_state=0)
    kmeans_test.fit(x_digits)
    inertias.append(kmeans_test.inertia_)

# Plot the elbow curve
plt.figure(figsize=(12, 6))
plt.plot(n_clusters_range, inertias, 'bo-', linewidth=2, markersize=4)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)

# Highlight k=10
plt.axvline(x=10, color='red', linestyle='--', label='k=10', alpha=0.7)
plt.legend()
plt.show()

print(f"Inertia at k=10: {inertias[9]:.2f}")