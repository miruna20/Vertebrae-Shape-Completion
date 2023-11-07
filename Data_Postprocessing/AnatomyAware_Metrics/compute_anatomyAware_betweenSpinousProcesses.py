import open3d as o3d
import numpy as np
import os
import argparse
import csv

def read_point_cloud_from_file(file_path):
    """
    Summary:
    This function reads txt file and returns a list of points
    Parameters:
    file_path: txt file with one point per line, exported from ImFusion Suite
    an example of a line in the txt file [703.906,248.236,191.862]
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

def point_to_line_distance(point, line_point, line_direction):
    """
    Compute the point to line distance between point and the line represented by a line point and a line direction
    """
    vector_point_to_line = point - line_point
    distance = np.linalg.norm(np.cross(vector_point_to_line, line_direction)) / np.linalg.norm(line_direction)
    return distance

def visualize_line(pointset1,line_dir, line_point , pointset2):
    """
    Visualize a line represented by a point along the line and a line direction together with 2 pointsets
    """
    # create pcd from the pointset1 so that we can visualize it
    pointset1_pcd = o3d.geometry.PointCloud()
    pointset1_pcd.points = o3d.utility.Vector3dVector(np.asarray(pointset1))

    pointset2_pcd = o3d.geometry.PointCloud()
    pointset2_pcd.points = o3d.utility.Vector3dVector(np.asarray(pointset2))

    # Create points for the line (start and end points)
    line_start = line_point - 100 * line_dir  # Extend the line for visualization
    line_end = line_point + 100 * line_dir

    # Create a LineSet geometry
    line_set = o3d.geometry.LineSet()

    # Add points to the LineSet
    line_set.points = o3d.utility.Vector3dVector(np.array([line_start, line_end]))

    # Add a line connecting the points
    lines = [[0, 1]]
    line_set.lines = o3d.utility.Vector2iVector(lines)

    o3d.visualization.draw([pointset1_pcd, line_set, pointset2_pcd])


def compute_dist_points_to_line_between_two_pointsets(points_set1, points_set2):
    """
    Compute distances between all points in point_set1 and the line represented by points_set2
    Take into consideration shifts along the direction of the line.
    """

    centroid = np.mean(points_set2, axis=0)
    uu, dd, vv = np.linalg.svd(points_set2 - centroid)
    line_direction = vv[0]
    linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]
    linepts += centroid

    min_y_point = points_set2[np.argmin(points_set2[:, 1])]
    max_y_point = points_set2[np.argmax(points_set2[:, 1])]

    # this works only because we know that the vertebrae are aligned along the y axis
    # TODO Make this applicable for all line_directions
    distances = []
    for point in points_set1:
        if point[1] < min_y_point[1]:
            distances.append(np.linalg.norm(point - min_y_point))
        elif point[1] > max_y_point[1]:
            distances.append(np.linalg.norm(max_y_point-point))
        else:
            distances.append(point_to_line_distance(point,linepts[0], line_direction))


    #visualize_line(points_set1, line_direction,linepts[0], points_set2)
    return distances


def compute_point_to_line_between_SP(SP_input, SP_completion):
    """
    Compute mean squared distance between symmetric point to line distances between the input and completion
    """
    """
    Given two sets of points FJ_GT and FJ_completion which annotate the upper parts of the spinous processes
    we want to measure the distance in between these two lines

    Strategy: For each point in FJ_GT compute the distance between it and a line fit through the points in
    FJ_completion. Do the same the other way around. Obtain two distance vectors dists1 and dists2

    Compute the mean of square distances.

    """
    # read the points from the txt files
    SP_input_points = np.array(read_point_cloud_from_file(SP_input))
    SP_completion_points = np.array(read_point_cloud_from_file(SP_completion))

    dists1 = compute_dist_points_to_line_between_two_pointsets(SP_input_points,SP_completion_points)
    dists2 = compute_dist_points_to_line_between_two_pointsets(SP_completion_points,SP_input_points)

    #point_to_line_dist = np.mean(dists)
    point_to_line_dist = np.mean(np.square(dists1) + np.square(dists2))

    return point_to_line_dist

def compute_CD_between_SP(SP_input, SP_completion):
    """
    Compute CD between the two point sets
    """
    # load point clouds from the txt files

    SP_input_points = read_point_cloud_from_file(SP_input)
    SP_input_pcd = o3d.geometry.PointCloud()
    SP_input_pcd.points = o3d.utility.Vector3dVector(np.asarray(SP_input_points))


    SP_completion_points = read_point_cloud_from_file(SP_completion)
    SP_completion_pcd = o3d.geometry.PointCloud()
    SP_completion_pcd.points = o3d.utility.Vector3dVector(np.asarray(SP_completion_points))
    red_color = np.array([1, 0, 0])  # RGB values for red
    colors = np.tile(red_color, (len(SP_completion_points), 1))  # Repeat red_color for all points
    SP_completion_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(os.path.join(os.path.dirname(SP_input),os.path.basename(SP_input) + ".pcd"), SP_input_pcd)
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(SP_completion),os.path.basename(SP_completion) + ".pcd"), SP_completion_pcd)

    # compute the CD metric
    dists1 = SP_input_pcd.compute_point_cloud_distance(SP_completion_pcd)
    dists2 = SP_completion_pcd.compute_point_cloud_distance(SP_input_pcd)

    #o3d.visualization.draw([SP_input_pcd, SP_completion_pcd])

    chamfer_distance = np.mean(np.square(dists1) + np.square(dists2))
    return chamfer_distance



"""
- This script does the following:
    -   Reads one **Input Centerline of the Spinous Process (from the segmentation)** and one **Completion Centerline of the Spinous Process**, both represented by 10 points 
    (these need to be previously manually marked)
    -   Compute for this pair the CD distance 
    -   Compute a point to line distance between the two point sets. This point to line takes into consideration shifts between the two lines along the line axis
    -   These are the anatomy-aware metrics for the spinous process
"""

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Compute anatomy aware metric for spinous processes ")
    arg_parser.add_argument(
        "--list_txt_files_spinousProcesses",
        required=True,
        dest="spinous_processes",
        help="Txt file with a list of names of the GT and completion pcds and sphere coordinates files"
    )

    args = arg_parser.parse_args()
    spinous_process_list = args.spinous_processes

    paths = []
    with open(spinous_process_list, 'r') as file:
        for pcd_path in file:
            paths.append(pcd_path.replace("\n", ""))

        # prepare table where to write the results from the anatomy aware metric
    header = ['Name', 'CD SP', 'PointToLine SP' ]
    csv_file = os.path.join(os.path.dirname(spinous_process_list), "results_anatomy_aware.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # for each pair of GT and completion
        for index in range(0, len(paths), 2):

            spinous_process_input = paths[index]
            spinous_process_completion = paths[index + 1]
            print(spinous_process_input)
            if not spinous_process_input.endswith('.txt'):
                raise Exception("Path does not end in .txt: " + spinous_process_input)

            if not spinous_process_completion.endswith('.txt'):
                raise Exception("Path does not end in .txt: " + spinous_process_completion)

            cd = compute_CD_between_SP(spinous_process_input, spinous_process_completion)
            point_to_line = compute_point_to_line_between_SP(spinous_process_input,spinous_process_completion )
            print("Chamfer distance is: " + str(cd))
            print("Point to line distance is: " + str(point_to_line))

            writer.writerow([os.path.basename(spinous_process_input)[:-4], str(cd),str(point_to_line)])
