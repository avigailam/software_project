import sys
import math
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf

# --- K-means Functions (From your HW1) ---

def euclidean_distance(p1, p2):
    d = len(p1)
    distance_sum = 0
    for i in range(d):
        distance_sum += (p1[i] - p2[i])**2
    return math.sqrt(distance_sum)

def initialize_centroids(data_points, k):
    centroids = []
    for i in range(k):
        centroids.append(data_points[i][:])
    return centroids

def assign_clusters(data_points, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data_points:
        min_distance = float('inf')
        min_index = 0
        for i in range(len(centroids)):
            distance = euclidean_distance(point, centroids[i])
            if distance < min_distance :
                min_distance = distance
                min_index = i
        clusters[min_index].append(point)
    return clusters

def update_centroids(clusters, k, dimension, first_point):
    new_centroids = [[0]*dimension for i in range(k)]
    for i in range(k):
        num_points = len(clusters[i])
        if num_points == 0:
            new_centroids[i] = [x for x in first_point]
            continue
        for point in clusters[i]:
            for j in range(dimension):
                new_centroids[i][j] += point[j]
        for j in range(dimension):
            new_centroids[i][j] /= num_points
    return new_centroids

def get_kmeans_labels(data_points, centroids):
    labels = []
    for point in data_points:
        min_distance = float('inf')
        min_index = 0
        for i in range(len(centroids)):
            distance = euclidean_distance(point, centroids[i])
            if distance < min_distance:
                min_distance = distance
                min_index = i
        labels.append(min_index)
    return labels

def run_kmeans(points, k):
    epsilon = 1e-4
    max_iter = 300
    dimension = len(points[0])
    
    centroids = initialize_centroids(points, k)
    first_point_in_data = points[0]
    
    for iteration_num in range(max_iter):
        old_centroids = centroids
        clusters = assign_clusters(points, old_centroids)
        new_centroids = update_centroids(clusters, k, dimension, first_point_in_data)
        centroids = new_centroids

        converged = True
        for i in range(k):
            delta_i = euclidean_distance(new_centroids[i], old_centroids[i])
            if delta_i >= epsilon:
                converged = False
                break
        if converged:
            break
            
    return centroids


def data_reader(file_name):
    """
    Reads the data points from the specified file and returns them as a list of lists.
    
    Args:
        file_name: The name of the input file containing the data points.
    Returns:
        Tuple of (data_points, n, d).
    """
    try:
        data_points = []
        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_points.append([float(x) for x in line.split(',')])
        if len(data_points) == 0:
            error_and_exit()
        d = len(data_points[0])
        for row in data_points:
            if len(row) != d:
                error_and_exit()
        return data_points, len(data_points), d
    except Exception:
        error_and_exit()

def symnmf_builder(data_points, n, d, k):
    """
    Builds the SymNMF result using the symnmf module and returns labels.
    
    Args:
        data_points: A list of lists representing the data matrix
        n: Number of data points (rows)
        d: Number of dimensions (columns)
        k: Number of clusters for symnmf
    Returns:
        symnmf_labels: Cluster labels from the final H matrix.
    """
    try:
        W = symnmf.norm(data_points, n, d)
        np.random.seed(1234)
        m = np.mean(W)
        high_bound = 2 * np.sqrt(m / k)
        H_init = np.random.uniform(low=0.0, high=high_bound, size=(n, k)).tolist()
        H_final = symnmf.symnmf(W, H_init, n, k)
        H_final_np = np.array(H_final)
        symnmf_labels = np.argmax(H_final_np, axis=1)
        return symnmf_labels
    except Exception:
        error_and_exit()

def kmeaner(data_points, k):
    """
    Runs K-means and returns labels for each point.
    
    Args:
        data_points: A list of lists representing the data matrix
        k: Number of clusters for k-means
    Returns:
        kmeans_labels: Cluster labels from K-means.
    """
    try:
        final_centroids = run_kmeans(data_points, k)
        kmeans_labels = get_kmeans_labels(data_points, final_centroids)
        return kmeans_labels
    except Exception:
        error_and_exit()

def eval_output(data_points, symnmf_labels, kmeans_labels):
    """
    Evaluates the results of SymNMF and K-means using silhouette score.
    
    Args:
        data_points: A list of lists representing the data matrix
        symnmf_labels: Labels returned by SymNMF
        kmeans_labels: Labels returned by K-means
    """
    try:
        X_np = np.array(data_points)
        score_nmf = silhouette_score(X_np, symnmf_labels)
        score_kmeans = silhouette_score(X_np, kmeans_labels)
        print(f"nmf: {score_nmf:.4f}")
        print(f"kmeans: {score_kmeans:.4f}")
    except Exception:
        error_and_exit()

# --- Main Analysis Script ---

def error_and_exit():
    print("An Error Has Occurred")
    sys.exit(1)

def main():
    if len(sys.argv) != 3:
        error_and_exit()

    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
    except ValueError:
        error_and_exit()

    data_points, n, d = data_reader(file_name)
    symnmf_labels = symnmf_builder(data_points, n, d, k)
    kmeans_labels = kmeaner(data_points, k)
    eval_output(data_points, symnmf_labels, kmeans_labels)

if __name__ == "__main__":
    main()