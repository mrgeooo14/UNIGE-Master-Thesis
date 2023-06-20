import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering

##### Dimensionality reduction for our word embedding vectors down to 2D using TSNE
##### The x_vals and y_vals store the x and y values for each word embedding, respectively
##### The labels variable contains the word associated to a given point
def reduce_dimensions_tsne(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components = num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels    
    
##### Current_Point and annotations are tuples that contain the X and Y coordinate
def check_overlap(current_point, annotations, overlap = False):
    for annotation in annotations:
        if ((np.abs(current_point[0] - annotation[0]) <= 1.5) and (np.abs(current_point[1] - annotation[1]) <= 1.5)):
            overlap = True
            break
    return overlap

##### k-Means clustering embeddings to center clusters
def kmeans_centers(x_values, y_values):
    kmeans = KMeans(n_clusters = 10)
    kmeans.fit(np.array([x_values, y_values]).T)
    labels = kmeans.predict(np.array([x_values, y_values]).T)
    centers = kmeans.cluster_centers_
    return labels, centers


##### Spectral clustering embeddings to a projection of the normalized Laplacian (nearest neighbors)
def nn_spectral_clustering(x_values, y_values):
    spectral = SpectralClustering(n_clusters = 10, affinity='nearest_neighbors', random_state = 0).fit(np.array([x_values, y_values]).T)
    labels = spectral.labels_
    return labels
