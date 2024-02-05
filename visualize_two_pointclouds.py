import open3d as o3d
import numpy as np


def load_ply(file_path, color):
    # Load PLY file
    ply_cloud = o3d.io.read_point_cloud(file_path)

    # Create a color array for the entire point cloud
    colors = np.array([color] * len(ply_cloud.points))
    ply_cloud.colors = o3d.utility.Vector3dVector(colors)

    return ply_cloud


if __name__ == "__main__":
    # Replace these file paths with the paths to your PLY files
    ply_file_path_1 = '/home/miruna20/Documents/01_PhD/projects/IPCAI/PCN/PCN_vertebrae_4096dense_1024coarse/patient8/L2/output/001.ply'
    ply_file_path_2 = '/home/miruna20/Documents/Thesis/ResultsExperiments/lumbar_vertebrae_from_US/new_pipeline/4096/complete_pipeline/inference/patient8/with_filter/patient8_ct_verLev22_3_GT.ply'

    # Load PLY files
    ply_cloud_1 = load_ply(ply_file_path_1, color=[0, 0, 1])  # Blue
    ply_cloud_2 = load_ply(ply_file_path_2, color=[0, 1, 0])  # Green

    # Merge point clouds
    merged_cloud = ply_cloud_1 + ply_cloud_2

    # Visualize the merged point cloud
    o3d.visualization.draw_geometries([merged_cloud])