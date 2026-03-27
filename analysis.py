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

    # Read data points from the file
    try:
        data_points = []
        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_points.append([float(x) for x in line.split(',')])
        n = len(data_points)
        if n == 0:
            error_and_exit()
        d = len(data_points[0])
    except Exception:
        error_and_exit()

    # 3. --- SymNMF ---
    try:
        # הקריאות ל-C מתוקנות כאן עם n ו-d
        W = symnmf.norm(data_points, n, d)
        
        np.random.seed(1234)
        m = np.mean(W)
        high_bound = 2 * np.sqrt(m / k)
        H_init = np.random.uniform(low=0.0, high=high_bound, size=(n, k)).tolist()
        
        H_final = symnmf.symnmf(W, H_init, n, k)
        
        H_final_np = np.array(H_final)
        symnmf_labels = np.argmax(H_final_np, axis=1)
        
    except Exception:
        error_and_exit()

    # 4. --- K-means ---
    try:
        final_centroids = run_kmeans(data_points, k)
        kmeans_labels = get_kmeans_labels(data_points, final_centroids)
    except Exception:
        error_and_exit()

    # 5. --- Evaluation & Output ---
    try:
        X_np = np.array(data_points)
        score_nmf = silhouette_score(X_np, symnmf_labels)
        score_kmeans = silhouette_score(X_np, kmeans_labels)

        print(f"nmf: {score_nmf:.4f}")
        print(f"kmeans: {score_kmeans:.4f}")
        
    except Exception:
        error_and_exit()

if __name__ == "__main__":
    main()