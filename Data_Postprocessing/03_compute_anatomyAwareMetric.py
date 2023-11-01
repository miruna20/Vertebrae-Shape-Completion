import open3d as o3d
import numpy as np
import os
import math
import argparse
import csv

"""
- Write a script that does the following
    -   Reads one **GT PCD** and one **completion PCD** as well as the corresponding txt files that contain the sphere coordinates
    -   Get all points within the GT PCD that belong to the annotated GT sphere
    -   Get all points within the completion PCD that belong to the annotated completion sphere
    -   Compute point to point metrics between these two (start with computing the chamfer distance, later also EMD)
    -   These metrics will be the anatomy aware ones! 😎
"""

def parse_sphere_txt_file(sphere_path):
    """
    From a txt file containing the center and outside point of a sphere get these 2 3D points
    """
    with open(sphere_path, 'r') as file:
        lines = file.readlines()

    center_point = np.array([float(value) for value in lines[0].replace("[", "").replace("]", "").replace("\n", "").split(',')])
    outside_point = np.array([float(value) for value in lines[1].replace("[", "").replace("]", "").replace("\n", "").split(',')])

    return center_point, outside_point

def get_PCD_points_within_sphere(pcd, sphere_center, sphere_outside_point):
    print("GT points within sphere")
    radius = math.sqrt((sphere_outside_point[0] - sphere_center[0])**2 + (sphere_outside_point[1] - sphere_center[1])**2 + (sphere_outside_point[2] - sphere_outside_point[2])**2)
    points = np.asarray(pcd.points)

    # calculate distances from the target coordinate
    distances = np.linalg.norm(points - sphere_center, axis=1)

    # find points within the specified radius
    points_within_radius = points[distances < radius]

    # create a new point cloud from the filtered points
    landmark_pcd = o3d.geometry.PointCloud()
    landmark_pcd.points = o3d.utility.Vector3dVector(points_within_radius)

    # Visualize
    #o3d.visualization.draw_geometries([landmark_pcd])
    return landmark_pcd


def compute_anatomyAwareCD(GT_pcd_path, completion_pcd_path, GT_sphere_path, completion_sphere_path):
    # GT_pcd and completion_pcd are open3d point clouds
    # GT_sphere and completion_sphere are txt files representing sphere coordinates and have been obtained
    # by marking the sphere on ImFusion

    # load GT pcd and completion pcd with open3d
    GT_pcd = o3d.io.read_point_cloud(GT_pcd_path)
    completion_pcd = o3d.io.read_point_cloud(completion_pcd_path)

    # load txt files line by line knowing that the first line is the center point and the second line is a point
    # on the outside radius of the sphere
    center_sphere_GT,outsidePoint_sphere_GT = parse_sphere_txt_file(GT_sphere_path)
    center_sphere_completion, outsidePoint_sphere_completion = parse_sphere_txt_file(completion_sphere_path)

    # get all points from the GT_pcd that are within a radius r of centroid of the GT sphere
    GT_landmark_pcd = get_PCD_points_within_sphere(GT_pcd,center_sphere_GT,outsidePoint_sphere_GT)
    completion_landmark_pcd = get_PCD_points_within_sphere(completion_pcd,center_sphere_completion,outsidePoint_sphere_completion)

    # compute the CD metric
    dists1 = GT_landmark_pcd.compute_point_cloud_distance(completion_landmark_pcd)
    dists2 = completion_landmark_pcd.compute_point_cloud_distance(GT_landmark_pcd)

    chamfer_distance = np.mean(np.square(dists1) + np.square(dists2))

    return chamfer_distance

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Compute anatomy aware metric")

    arg_parser.add_argument(
        "--list_vertebrae_spheres",
        required=True,
        dest="vert_list",
        help="Txt file with a list of names of the GT and completion pcds and sphere coordinates files"
    )


    args = arg_parser.parse_args()
    vert_list = args.vert_list

    paths = []
    with open(vert_list, 'r') as file:
        for pcd_path in file:
            paths.append(pcd_path.replace("\n", ""))

    # prepare table where to write the results from the anatomy aware metric
    header = ['Name','Anatomy-aware CD']
    csv_file = os.path.join(os.path.dirname(vert_list),"results_anatomy_aware.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # for each pair of GT and completion
        for index in range(0, len(paths), 4):
            GT_pcd_path = paths[index]
            GT_sphere_path = paths[index+1]
            completion_pcd_path = paths[index+2]
            completion_sphere_path = paths[index+3]

            if not GT_pcd_path.endswith('.pcd'):
                raise Exception("Path does not end in .pcd: " + GT_pcd_path)

            if not completion_pcd_path.endswith('.pcd'):
                raise Exception("Path does not end in .pcd: " + completion_pcd_path)

            if not GT_sphere_path.endswith('.txt'):
                raise Exception("Path does not end in .txt: " + GT_sphere_path)

            if not completion_sphere_path.endswith('.txt'):
                raise Exception("Path does not end in .txt: " + completion_sphere_path)

            cd = compute_anatomyAwareCD(GT_pcd_path, completion_pcd_path,GT_sphere_path,completion_sphere_path)
            print("Chamfer distance is: " + str(cd))

            writer.writerow([os.path.basename(GT_pcd_path)[:-4],str(cd)])


