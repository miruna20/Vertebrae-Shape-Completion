import open3d as o3d
import numpy as np
import os
import argparse
import csv
import fps

def compute_CD_between_TP(TP_input, TP_completion):
    """
    Compute CD between two point clouds
    """
    # read pcd of TP
    TP_input_pcd = o3d.io.read_point_cloud(TP_input)
    TP_completion_pcd = o3d.io.read_point_cloud(TP_completion)

    # the point clouds do not always have the same number of points
    if(len(TP_input_pcd.points) < len(TP_completion_pcd.points)):
        # sample TP_completion_pcd with FPS
        sampled_TP_completion_points = fps.fps_points(np.asarray(TP_completion_pcd.points), len(TP_input_pcd.points))
        TP_completion_pcd.points = o3d.utility.Vector3dVector(sampled_TP_completion_points)
    elif len(TP_input_pcd.points) > len(TP_completion_pcd.points):
        # sample TP_input_pcd with FPS
        sampled_TP_input_points = fps.fps_points(np.asarray(TP_input_pcd.points), len(TP_completion_pcd.points))
        TP_input_pcd.points = o3d.utility.Vector3dVector(sampled_TP_input_points)

    # compute the CD metric
    dists1 = TP_input_pcd.compute_point_cloud_distance(TP_completion_pcd)
    dists2 = TP_completion_pcd.compute_point_cloud_distance(TP_input_pcd)

    chamfer_distance = np.mean(np.square(dists1) + np.square(dists2))

    return chamfer_distance

"""
- This script does the following:
    -   Reads one **Input point cloud that represents one transverse process** and one **Completion point cloud that represents one transverse process**,
    -   Compute chamfer distance between these 2 point clouds 
"""

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Compute anatomy aware metric for transverse processes ")
    arg_parser.add_argument(
        "--list_txt_files_transverseProcesses",
        required=True,
        dest="transverse_processes",
        help="Txt file with a list TP of input and of completion"
    )

    args = arg_parser.parse_args()
    transverse_process_list = args.transverse_processes

    paths = []
    with open(transverse_process_list, 'r') as file:
        for pcd_path in file:
            paths.append(pcd_path.replace("\n", ""))

        # prepare table where to write the results from the anatomy aware metric
    header = ['Name', 'CD TP' ]
    csv_file = os.path.join(os.path.dirname(transverse_process_list), "results_TP_anatomy_aware.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # for each pair of GT and completion
        for index in range(0, len(paths), 2):

            TP_input = paths[index]
            TP_completion = paths[index + 1]
            print(TP_input)
            if not TP_input.endswith('.pcd'):
                raise Exception("Path does not end in .pcd: " + TP_input)

            if not TP_completion.endswith('.pcd'):
                raise Exception("Path does not end in .pcd: " + TP_completion)

            cd = compute_CD_between_TP(TP_input, TP_completion)
            print("Chamfer distance is: " + str(cd))

            writer.writerow([os.path.basename(TP_input)[:-4], str(cd)])
