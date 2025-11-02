import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_dilation, distance_transform_edt, find_objects
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

# --- KONFIGURASI PIPELINE MORFOMETRIK ---

VOXEL_SIZE = 0.25           # Resolusi grid
DILATION_ITERATIONS = 2     # "Lem" kuat untuk menyambungkan fragmen internal
THRESHOLD_PERCENTAGE = 0.3  # Ambil 30% area tertebal sebagai kandidat marker
MIN_MARKER_SIZE_VOXELS = 5  # Saring marker yang lebih kecil dari 5 voxel (noise)
# ----------------------------------------

def voxelize_point_cloud(points, voxel_size):
    """Mengubah point cloud menjadi 3D voxel grid biner beserta batasnya."""
    if points.shape[0] == 0:
        return None, None
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int) + 1
    voxel_grid = np.zeros(dims, dtype=bool)
    indices = np.floor((points - min_bound) / voxel_size).astype(int)
    indices = np.clip(indices, 0, np.array(dims) - 1)
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return voxel_grid, min_bound

def run_morphometry_pipeline(points, labels):
    """
    Fungsi utama untuk menjalankan seluruh pipeline ekstraksi morfometrik.
    Menerima 'points' dan 'labels' dari file .npz, mengembalikan DataFrame hasil.
    """
    # 1. Isolasi titik-titik Rugae (label=2)
    rugae_points = points[labels == 2]
    if rugae_points.shape[0] < 50: # Butuh cukup titik
        return pd.DataFrame({"Info": ["Tidak cukup data rugae untuk dianalisis."]})

    # 2. Voxelization dan Dilasi (Lem Kuat)
    rugae_voxel_grid, min_bound = voxelize_point_cloud(rugae_points, VOXEL_SIZE)
    dilated_grid = binary_dilation(rugae_voxel_grid, iterations=DILATION_ITERATIONS)
    
    # 3. Hitung Peta Jarak untuk Watershed
    distance_map = distance_transform_edt(dilated_grid)
    max_dist_voxels = distance_map.max()
    
    # 4. Temukan Marker Awal dengan Threshold Rendah
    simple_threshold = max_dist_voxels * THRESHOLD_PERCENTAGE
    initial_markers_mask = distance_map > simple_threshold
    initial_markers, num_initial = label(initial_markers_mask)
    
    # 5. Filter Marker berdasarkan Ukuran Minimum
    final_markers_mask = np.zeros_like(initial_markers_mask, dtype=bool)
    slices = find_objects(initial_markers)
    for i, slc in enumerate(slices):
        if slc is None: continue
        marker_id = i + 1
        marker_size = np.sum(initial_markers[slc] == marker_id)
        if marker_size >= MIN_MARKER_SIZE_VOXELS:
            final_markers_mask[initial_markers == marker_id] = True
            
    # 6. Beri Label pada Marker Final
    final_markers, num_final_markers = label(final_markers_mask)

    # 7. Jalankan Watershed
    if num_final_markers > 0:
        labels_ws = watershed(-distance_map, final_markers, mask=dilated_grid)
        num_features = len(np.unique(labels_ws)) - 1
    else:
        return pd.DataFrame({"Info": ["Tidak ada marker valid ditemukan, proses dihentikan."]})

    if num_features == 0:
        return pd.DataFrame({"Info": ["Tidak ada rugae yang terdeteksi setelah watershed."]})
        
    # 8. Hitung Skeleton dan Panjang untuk setiap rugae
    results = []
    for i in range(1, num_features + 1):
        single_rugae_grid = (labels_ws == i)
        skeleton = skeletonize(single_rugae_grid)
        length_mm = np.sum(skeleton) * VOXEL_SIZE
        results.append({"Rugae ID": i, "Estimasi Panjang (mm)": round(length_mm, 2)})
        
    return pd.DataFrame(results)