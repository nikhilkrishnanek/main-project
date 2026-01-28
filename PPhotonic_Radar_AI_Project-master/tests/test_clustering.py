import numpy as np
from signal_processing.detection import cluster_detections

def test_clustering_single_peak():
    # 5x5 map with a 3x3 block of "detections" centered at (2,2)
    det_map = np.zeros((5, 5), dtype=bool)
    det_map[1:4, 1:4] = True
    
    # Power map where (2,2) is the true peak
    rd_map = np.zeros((5, 5))
    rd_map[1:4, 1:4] = 0.5
    rd_map[2, 2] = 1.0 # Peak
    
    centroids = cluster_detections(det_map, rd_map)
    
    assert len(centroids) == 1
    assert centroids[0] == (2, 2)
    print("✅ Single peak clustering PASSED")

def test_clustering_multiple_targets():
    det_map = np.zeros((10, 10), dtype=bool)
    # Cluster 1
    det_map[1:3, 1:3] = True
    # Cluster 2
    det_map[7:9, 7:9] = True
    
    rd_map = np.zeros((10, 10))
    rd_map[1, 1] = 1.0 # Max of cluster 1
    rd_map[8, 8] = 2.0 # Max of cluster 2
    
    centroids = cluster_detections(det_map, rd_map)
    
    assert len(centroids) == 2
    assert (1, 1) in centroids
    assert (8, 8) in centroids
    print("✅ Multiple target clustering PASSED")

if __name__ == "__main__":
    test_clustering_single_peak()
    test_clustering_multiple_targets()
