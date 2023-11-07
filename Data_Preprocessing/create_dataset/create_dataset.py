import argparse
import os.path
import re
import open3d as o3d
import numpy as np
import fps
import h5py
import collections
import math

"""
Pair one partial shape with a complete shape
"""

def extractLabel(vert_path):
    """
    Extract the label corresponding to the vertebrae level from the path of the verterbae
    :param vert_path:
    :return:
    """
    # get the base name of the file from the path
    basename = os.path.basename(vert_path)
    match = re.search(r'verLev(\d+)', basename)
    label = match.group(1)
    return int(label)

def sample_partial_pcd(partial_pcd_path, nr_points):
    """
    Sample a certain number of points from a point cloud with furthest point sampling
    """

    partial_pcd = o3d.io.read_point_cloud(partial_pcd_path)
    nr_points_in_partial_pcd = np.asarray(partial_pcd.points).shape[0]

    if (nr_points_in_partial_pcd >= nr_points):
        sampled_partial_pcd = fps.fps_points(np.asarray(partial_pcd.points), num_samples=nr_points)
    else:
        print("PCD with less than " + str(nr_points) + "points" + str(partial_pcd_path))
        return 0, []

    return sampled_partial_pcd

def sample_complete_mesh(complete_mesh_path, nr_points):
    """
    Generate point cloud with a certain number of points by applying poisson disk sampling on the mesh
    """
    completeVertebra = o3d.io.read_triangle_mesh(complete_mesh_path)
    complete_pcd = o3d.geometry.TriangleMesh.sample_points_poisson_disk(completeVertebra, nr_points)
    complete_pcd_points = np.asarray(complete_pcd.points)
    return complete_pcd_points

def get_nr_samples_per_class(labels, nr_partial_pcds_per_shape=16):
    # the labels need to start with 0 (make sure that if you have labels for lumbar vertebrae you substract minimal label

    nr_samples_per_class = collections.Counter(labels)
    ordered_dict = collections.OrderedDict(sorted(nr_samples_per_class.items()))

    nr_samples_per_class_as_list = []
    for key in ordered_dict.keys():
        nr_samples_per_class_as_list.append(math.floor(ordered_dict[key] / nr_partial_pcds_per_shape))

    return np.asarray(nr_samples_per_class_as_list)


def saveToH5(file_name, partial_pcds_stacked, complete_pcds_stacked, labels, datasets_ids, samples_per_class=16):
    # save the dataset in a .h5 file for VRCNet
    vertebrae_file = h5py.File(file_name, "w")
    dset_incompletepcds = vertebrae_file.create_dataset("incomplete_pcds", data=partial_pcds_stacked)
    dset_completepcds = vertebrae_file.create_dataset("complete_pcds", data=complete_pcds_stacked)
    dset_labels = vertebrae_file.create_dataset("labels", data=labels)
    dset_ids = vertebrae_file.create_dataset("datasets_ids", data=datasets_ids)
    number_per_class = get_nr_samples_per_class(labels, samples_per_class)
    dset_number_per_class = vertebrae_file.create_dataset("number_per_class", data=number_per_class)


def create_dataset(vert_list, nr_points):

    # get all_labels and the minimum one
    all_labels = [extractLabel(vert_path) for vert_path in vert_list]
    min_label = np.min(np.asarray(all_labels))

    labels = []
    complete_pcds_all_vertebrae = []
    partial_pcds_all_vertebrae = []
    dataset_ids = []

    # for each path in the list
    for index in range(0,len(vert_list),2):
        print("curr_index: " + str(index))
        partial_pcd_path = vert_list[index]
        if not partial_pcd_path.endswith('.pcd'):
            raise Exception("Path does not end in .pcd: " + partial_pcd_path)

        complete_obj_path = vert_list[index+1]
        if not complete_obj_path.endswith('.obj'):
            raise Exception("Path does not end in .obj: " + complete_obj_path)

        incomplete_pcd = sample_partial_pcd(partial_pcd_path,nr_points)
        complete_pcd = sample_complete_mesh(complete_obj_path, nr_points)

        # in case the partial_pcd had initially less points than the number we want to sample
        if len(incomplete_pcd) == 0:
            continue

        # get label
        label_normalized = all_labels[index] - min_label

        # create lists for dataset
        partial_pcds_all_vertebrae.append(incomplete_pcd)
        complete_pcds_all_vertebrae.append(complete_pcd)
        labels.extend([label_normalized for j in range(0, 1)])
        dataset_ids.append((str(os.path.basename(partial_pcd_path)[:-4])).encode("ascii"))

    # stack all lists
    stacked_partial_pcds = np.stack(partial_pcds_all_vertebrae, axis=0)
    stacked_complete_pcds = np.stack(complete_pcds_all_vertebrae, axis=0)
    stacked_dataset_ids = np.stack(dataset_ids, axis=0)
    labels_array = np.asarray(labels)

    # save to H5
    saveToH5("dataset.h5", partial_pcds_stacked=stacked_partial_pcds, complete_pcds_stacked=stacked_complete_pcds,
             labels=labels_array, datasets_ids=stacked_dataset_ids, samples_per_class=1)


if __name__ == "__main__":
    # Example run: python3 create_dataset.py --vertebrae_list example/vert_list.txt
    # generates example dataset.h5

    arg_parser = argparse.ArgumentParser(description="Create a dataset for completion training")

    arg_parser.add_argument(
        "--vertebrae_list",
        required=True,
        dest="vert_list",
        help="Txt file with a list of paths of partial point clouds and obj files of vertebrae"
    )

    args = arg_parser.parse_args()
    vert_list = args.vert_list

    nr_points = 4096

    vertebrae_paths = []
    with open(vert_list, 'r') as file:
        for pcd_path in file:
            vertebrae_paths.append(pcd_path.replace("\n", ""))

    create_dataset(vertebrae_paths,nr_points)




