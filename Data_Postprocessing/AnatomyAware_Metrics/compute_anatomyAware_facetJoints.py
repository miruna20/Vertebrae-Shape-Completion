import numpy as np
import os
import argparse
import csv

def read_sphere_from_txt(file_path):
    """
    Summary:
    This function reads txt file and returns a list of points
    Parameters:
    file_path: txt file with one point per line, one point is the center of the sphere, the second one point on the exterior of the sphere
    exported from ImFusion Suite an example of a line in the txt file [703.906,248.236,191.862]
    """
    points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Remove square parentheses and split the line by comma
            point_str = line.strip()[1:-1]
            coordinates = point_str.split(',')
            # Convert coordinates to float and create a tuple for each point
            point = (float(coordinates[0]), float(coordinates[1]), float(coordinates[2]))
            points.append(point)

    return points

def compute_dist_facet_joints(FJ_GT, FJ_completion):
    """
    Compute euclidian distance between the FJ_GT and the FJ_completion
    """

    # here the first point is the center and the second one is the radius of the sphere
    FJ_GT_points = read_sphere_from_txt(FJ_GT)
    FJ_completion_points = read_sphere_from_txt(FJ_completion)

    # compute the distance between the 2 spheres
    dist = np.linalg.norm(np.array(FJ_GT_points[0]) - np.array(FJ_completion_points[0]))

    return dist

"""
- This script does the following:
    -   Reads one **GT center point of the facet joints** and one **Completion center point of the facet joints**,
    -   Compute euclidian distance between these 2 points 
"""
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Compute anatomy aware metric for facet joints ")
    arg_parser.add_argument(
        "--list_txt_files_facetJoints",
        required=True,
        dest="facet_joints_spheres",
        help="Txt file with a list facet joints spheres of GT and completion "
    )

    args = arg_parser.parse_args()
    facet_joints_spheres_list = args.facet_joints_spheres

    paths = []
    with open(facet_joints_spheres_list, 'r') as file:
        for pcd_path in file:
            paths.append(pcd_path.replace("\n", ""))

        # prepare table where to write the results from the anatomy aware metric
    header = ['Name', 'Dist Facet Joints' ]
    csv_file = os.path.join(os.path.dirname(facet_joints_spheres_list), "results_facet_joints_anatomy_aware.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # for each pair of GT and completion
        for index in range(0, len(paths), 2):

            FJ_GT = paths[index]
            FJ_completion = paths[index + 1]
            print(FJ_GT)
            if not FJ_GT.endswith('.txt'):
                raise Exception("Path does not end in .txt: " + FJ_GT)

            if not FJ_completion.endswith('.txt'):
                raise Exception("Path does not end in .txt: " + FJ_completion)

            cd = compute_dist_facet_joints(FJ_GT, FJ_completion)
            print("Distance is: " + str(cd))

            writer.writerow([os.path.basename(FJ_GT)[:-4], str(cd)])
