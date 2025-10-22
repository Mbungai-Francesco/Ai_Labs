from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

x_digits, y_digits = load_digits(n_class=10, return_X_y=True)
print(f'Original features shape: {x_digits.shape}')
### - Estimate the Kmeans clustering.
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(x_digits)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

def plot_centroids(centroids):
    """
    Plot the centroids of the clusters.
    Parameters:
    - centroids (numpy.ndarray): Array of centroids.
    """
    fig = plt.figure(figsize=(10,5))

    for i,curcen in enumerate(centroids):

        plt.subplot(2, 5, i+1) # Suppose that you have ten centroids.
        im_cen = curcen.reshape((8,8))
        plt.imshow(im_cen, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])

    plt.show()

#plot_centroids(centers)

# This code allow you to select 10 random examples 
whichex = np.random.randint(low=0,high=100,size=1) 
X_samp = np.concatenate([x_digits[y_digits==i][whichex] for i in range(10)])

# You can take a look at the shape of the examples: 
X_samp.shape
print(X_samp.shape)

### - Use the transform method from the kmeans object on X_samp.
### You will obtain an array containing the distances to the centroids. 
### - Use the argmin method from numpy to generate an array containing the index corresponding to the closest centroid to the samples
### - Fetch the corresponding centroid in another array closest_centroids.
### - And finally calculate the distances of each samples to its closest centroid using np.min

distances = kmeans.transform(X_samp)
idx_closest_centroids = np.argmin(distances, axis=1)
closest_centroids= centers[idx_closest_centroids]
smallest_distances = np.min(distances, axis=1)

print(f'Distances array shape: {distances.shape}, should be (10,10).')
print(f'Indexes of closest centroids array shape {idx_closest_centroids.shape}, should be (10,).')
print(f'Closest centroids array shape {closest_centroids.shape}, should be (10,64).')
print(f'Smallest distances array shape {smallest_distances.shape}, should be (10,).')

### - Check that the indices of your closest centroids are the same than the ones obtained using the predict method.
print(f'Closest centroids according to your code\t: {idx_closest_centroids}')
print(f'Closest centroids according to sklearn\t\t: {kmeans.predict(X_samp)}')

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

#plot_digits_and_closest_centroids(X_samp, closest_centroids, smallest_distances)

# Visualize the elbow method with inertia
# Try different numbers of clusters and store inertias
n_clusters_range = range(1, 99)  # Réduit à 20 au lieu de 99 pour un temps d'exécution raisonnable
inertias = []

print("Calculating inertias for different numbers of clusters...")
for n_clusters in n_clusters_range:
    # Create and fit KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(x_digits)
    # Store the inertia
    inertias.append(kmeans.inertia_)
    if n_clusters % 5 == 0:  # Progress indication every 5 clusters
        print(f"Processed {n_clusters} clusters...")

def plot_elbow_method(n_clusters_range, inertias):
        # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, inertias, 'bo-')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude (Elbow method) pour KMeans sur digits')
    plt.grid(True)
    # Ajouter des annotations pour quelques points clés
    for k in [5, 10, 15, 20, 30, 50, 70, 90, 95]:
        plt.annotate(
            f'k={k}', 
            xy=(k, inertias[k-1]),
            xytext=(10, 10),
            textcoords='offset points'
        )
    plt.show()

    # Imprimer quelques valeurs clés pour analyse
    print("\nQuelques valeurs d'inertie :")
    for k in [5, 10, 15, 20, 30, 50, 70, 90, 95]:
        print(f"k={k}: inertie={inertias[k-1]:.2f}")

#plot_elbow_method(n_clusters_range, inertias)


# PCA
n_components = 16

pca = PCA(n_components=n_components)
x_digits_pcaV = pca.fit(x_digits)
componentsV =x_digits_pcaV.components_
#print(f'PCA V shape: {componentsV.shape}, should be ({n_components},64).')

def plot_pca_components(components):
    """
    Plot the components of the PCA.
    Parameters:
    - components (numpy.ndarray): Array of components.
    """
    fig, axis = plt.subplots(4, 4)
    for i, d in enumerate(components):
        ax = axis[i//4][i%4]
        ax.imshow(d.reshape((8, 8)), cmap=plt.cm.gray, vmin=np.min(components), vmax=np.max(components))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

#plot_pca_components(componentsV)

# Randomly select some examples.
whichex = np.random.randint(low=0, high=100, size=1) 
samples = list()
indexes = list()
for i in range(10):
    index = np.where(y_digits==i)[0][whichex]
    samples.append(x_digits[index])
    indexes.append(index)
X_samp = np.concatenate(samples)
indexes = np.array(indexes)

### TODO:
### - Generate the reconstructions array using the weights and the components.
### Recall that PCA is a matrix decomposition, hence the result of the decomposition may be retrive using matrix product.
weights = pca.transform(X_samp)
reconstructions = np.dot(weights, componentsV)
#print(f'Reconstructions shape: {reconstructions.shape}, should be (10,64).')

def plot_original_digits_and_pca_reconstruction(X_samp, reconstructions):
    """
    Plot the original digits and their PCA reconstructions.
    Parameters:
    - X_samp (numpy.ndarray): Array of original digits.
    - reconstructions (numpy.ndarray): Array of reconstructed digits.
    """
    plt.figure(figsize=(20,5))
    for plot_index,(digit,reconstruction) in enumerate(zip(X_samp,reconstructions)):
        plt.subplot(2,10,plot_index*2+1)

        plt.imshow(digit.reshape((8,8)),cmap=plt.cm.gray,vmin=x_digits.min(),vmax=x_digits.max())
        plt.xticks([])
        plt.yticks([])
        plt.title('$x$')

        plt.subplot(2,10,plot_index*2+2)
        plt.imshow(reconstruction.reshape((8,8)),cmap=plt.cm.gray,vmin=reconstructions.min(),vmax=reconstructions.max())
        plt.xticks([])
        plt.yticks([])
        error = np.sum((reconstruction-digit)**2)
        plt.title('${\~x}$, error %.2E' % error)

    plt.show()

#plot_original_digits_and_pca_reconstruction(X_samp,reconstructions)

# T-SNE
unsup = TSNE(random_state=0)
examples = unsup.fit_transform(x_digits)
plt.scatter(examples[:,0], examples[:,1], c=y_digits)
plt.colorbar()
plt.title("T-SNE visualization of digits dataset")
plt.show()