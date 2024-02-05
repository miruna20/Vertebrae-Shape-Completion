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

def sample_partial_pcd(partial_pcd, nr_points):
    """
    Sample a certain number of points from a point cloud with furthest point sampling
    """
    nr_points_in_partial_pcd = np.asarray(partial_pcd.points).shape[0]

    if (nr_points_in_partial_pcd >= nr_points):
        sampled_partial_pcd = fps.fps_points(np.asarray(partial_pcd.points), num_samples=nr_points)
    else:
        print("PCD with less than " + str(nr_points) + "points" + str(partial_pcd_path))
        return 0, []

    return sampled_partial_pcd

def sample_complete_mesh(completeVertebra, nr_points):
    """
    Generate point cloud with a certain number of points by applying poisson disk sampling on the mesh
    """
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

def load_corresponding_synth_template(filename_wo_ext):
    match = re.search(r'verLev(\d+)', filename_wo_ext)
    number = match.group(1)

    synth_template_path = os.path.join("synth_templates", "template_verLev" + number + ".pcd")
    synthetic_pcd = o3d.io.read_point_cloud(synth_template_path)
    return synthetic_pcd

def scale_to_unit_sphere(partial_pcd, complete_vert):
    #### scale to the unit sphere ####
    unit_sphere_size = 1
    bb_partial_pcd = partial_pcd.get_axis_aligned_bounding_box()
    # we know that the length between the two transverse processes will be along the x axis because we assume the alignment of the vert with the axes
    length_partial_pcd = bb_partial_pcd.get_max_bound()[0] - bb_partial_pcd.get_min_bound()[0]
    # + 20 here is just a padding to ensure that the full shape of the vertebra
    # will fit in the unit sp30here (which it won't without padding if the axis from arch to vert body
    # is longer than the one in between transverse process
    scaling_factor = unit_sphere_size / (length_partial_pcd + 30)

    partial_pcd.scale(scaling_factor, center=np.asarray([0, 0, 0]))
    complete_vert.scale(scaling_factor, center=np.asarray([0, 0, 0]))

    return partial_pcd,complete_vert,scaling_factor

def align_real_to_synthetic(real_pcd,complete_vert,synthetic_pcd):
    """
    Align the real input point cloud with a synthetic template of the same level
    """

    # match the centers of the 2 point clouds
    center_synthetic = np.asarray(synthetic_pcd.get_center())
    center_real = np.asarray(real_pcd.get_center())
    translation = center_synthetic - center_real

    real_pcd.translate(translation)
    complete_vert.translate(translation)

    # run ICP on them so that we get an aligned real dataset
    # Set the ICP convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=200)

    # Perform ICP registration
    reg_result = o3d.pipelines.registration.registration_icp(real_pcd, synthetic_pcd, 0.1, np.identity(4),
                                                   o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                             criteria)
    #o3d.io.write_point_cloud("/home/miruna20/Documents/PhD/PatientDataPreprocessing/Data/preProc_testing/real_pcd.pcd", real_pcd)

    real_pcd.transform(reg_result.transformation)
    complete_vert.transform(reg_result.transformation)

    return real_pcd,complete_vert, synthetic_pcd,translation,reg_result.transformation

def save_inverse_trafo(ICP_trafo, transl, scaling_factor, center, aligned_pcd,folder,filename_wo_ext):
    inverse_trafo = np.linalg.inv(ICP_trafo)

    transl_from_synth = np.array([[1, 0, 0, -transl[0]],
                                  [0, 1, 0, -transl[1]],
                                  [0, 0, 1, -transl[2]],
                                  [0, 0, 0, 1]])
    scale_up = np.array([[1 / scaling_factor, 0, 0, 0],
                         [0, 1 / scaling_factor, 0, 0],
                         [0, 0, 1 / scaling_factor, 0],
                         [0, 0, 0, 1]])
    trans_to_initial_pose = np.array([[1, 0, 0, center[0] - aligned_pcd.get_center()[0]],
                                      [0, 1, 0, center[1] - aligned_pcd.get_center()[1]],
                                      [0, 0, 1, center[2] - aligned_pcd.get_center()[2]],
                                      [0, 0, 0, 1]])
    # we want to first apply the inverse trafo and then the trafo scales and reverse (so we need to multiply in the reverse order)
    combined_trafo_inverse = trans_to_initial_pose @ scale_up @ transl_from_synth @ inverse_trafo
    # Test that the inverted transformation results in the original point cloud
    aligned_pcd.transform(combined_trafo_inverse)
    o3d.io.write_point_cloud(os.path.join(folder, filename_wo_ext + "_transformedback.pcd"),
                             aligned_pcd)

    # save the transformation matrix from centered to orig
    trafo_txt_file = os.path.join(folder, filename_wo_ext + "backward_transformed.txt")
    np.savetxt(trafo_txt_file, combined_trafo_inverse, fmt="%d")

def scale_and_align_to_synthetic(partial_pcd, complete_vert, folder, filename_wo_ext):
    # we have both partial and complete one here

    # find template
    synthetic_pcd = load_corresponding_synth_template(filename_wo_ext)

    #### move to center ####
    center = partial_pcd.get_center()
    partial_pcd.translate(-center)
    complete_vert.translate(-center)

    partial_pcd,complete_vert,scaling_factor = scale_to_unit_sphere(partial_pcd,complete_vert)

    #### register partial pcd to synthetic template pcd ####
    partial_pcd, complete_vert, synthetic_pcd, transl, ICP_trafo = align_real_to_synthetic(partial_pcd, complete_vert,synthetic_pcd)

    # save result in the same folder
    save_to_path = os.path.join(folder, filename_wo_ext + "_transformed.pcd")
    # o3d.visualization.draw_geometries([aligned_pcd])
    o3d.io.write_point_cloud(save_to_path, partial_pcd)
    o3d.io.write_triangle_mesh(save_to_path.replace(".pcd",".obj"),complete_vert)

    # save also the inverse transformation
    save_inverse_trafo(ICP_trafo,transl,scaling_factor,center,partial_pcd,folder,filename_wo_ext)

    return partial_pcd, complete_vert

def create_dataset(vert_list, nr_points,path_dataset):

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

        partial_pcd = o3d.io.read_point_cloud(partial_pcd_path)
        complete_vert = o3d.io.read_triangle_mesh(complete_obj_path)
        folder = os.path.dirname(partial_pcd_path)
        filename_wo_ext = os.path.basename(partial_pcd_path)[:-4]
        partial_pcd,complete_vert = scale_and_align_to_synthetic(partial_pcd,complete_vert,folder,filename_wo_ext)

        incomplete_pcd = sample_partial_pcd(partial_pcd,nr_points)
        complete_pcd = sample_complete_mesh(complete_vert, nr_points)

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
    saveToH5(path_dataset, partial_pcds_stacked=stacked_partial_pcds, complete_pcds_stacked=stacked_complete_pcds,
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

    path_dataset = os.path.join(os.path.dirname(vert_list), "dataset.h5")
    create_dataset(vertebrae_paths,nr_points,path_dataset)




