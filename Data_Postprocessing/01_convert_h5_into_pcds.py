import numpy as np
import h5py
import open3d as o3d
import os
import argparse

def get_filtered_pcd_from_numpy(np_array):
    """
    Create o3d pcd and then filter it
    """
    # create empty point cloud
    pcd = o3d.geometry.PointCloud()

    # add points to the point cloud
    pcd.points = o3d.utility.Vector3dVector(np_array)

    # filter pcd
    #filtered_np_array, _ = pcd.remove_radius_outlier(5, 0.05)
    return pcd

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Extract point clouds from h5 dataset")

    arg_parser.add_argument(
        "--results_dataset",
        required=False,
        dest="results_dataset",
        help="Path to the results dataset obtained from inference"
    )

    arg_parser.add_argument(
        "--inputs_dataset",
        required=False,
        dest="inputs_dataset",
        help="Path to the inference dataset"
    )

    arg_parser.add_argument(
        "--root_path_pcds",
        required=False,
        dest="root_path_pcds",
        help="Path where the resulting point clouds will be saved "
    )

    arg_parser.add_argument(
        "--root_path_trafos",
        required=False,
        dest="root_path_trafos",
        help="Root paths of txt files containing the transformations to the original pose of the vertebrae"
    )

    arg_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize point clouds "
    )
    args = arg_parser.parse_args()

    # this covers the case in which we have a lot of vertebrae in the results dataset and we only want to transform the
    # first 100 to point clouds
    nr_pcds = 100

    # read the h5 files of the datasets
    results_dataset = h5py.File(args.results_dataset, 'r')
    inputs_dataset = h5py.File(args.inputs_dataset, 'r')

    # get the numpy arrays
    inputs = np.array(inputs_dataset['incomplete_pcds'])
    names = np.array(inputs_dataset['datasets_ids'])
    results = np.array(results_dataset["results"][()])
    gts = np.array(results_dataset["gt"][()])

    os.makedirs(args.root_path_pcds, exist_ok=True)

    # however if we have a patient / phantom dataset with less than 5, then we can make the nr_pcds equal to the total
    # number of point clouds in the result file
    if(results.shape[0]<=5):
        nr_pcds = results.shape[0]

    for i in range(0, nr_pcds):
        filtered_input = get_filtered_pcd_from_numpy(inputs[i])
        filtered_gt = get_filtered_pcd_from_numpy(gts[i])
        filtered_result = get_filtered_pcd_from_numpy(results[i])

        if(args.visualize):
            filtered_input.paint_uniform_color([1, 0, 0])
            filtered_gt.paint_uniform_color([0, 1, 0])
            filtered_result.paint_uniform_color([0, 0, 1])

            o3d.visualization.draw_geometries([filtered_input,filtered_gt,filtered_result])
            o3d.visualization.draw_geometries([filtered_input,filtered_result])
            o3d.visualization.draw_geometries([filtered_input])

        curr_name = names[i].decode("utf-8")
        # Apply reverse transformation to get the completion point cloud to be in the original coordinate system
        # first find the corresponding txt file
        txt_file = os.path.join(args.root_path_trafos,curr_name + ".txt")

        # read the numpy from this txt file
        trafo = np.loadtxt(txt_file)

        # apply trafo to the completion point cloud
        filtered_input.transform(trafo)
        filtered_result.transform(trafo)

        # trafo_phantom_IFL
        #trafo_IFL = np.loadtxt("/home/miruna20/Documents/PhD/phantom_IFL/segm_with_CT_available/US/input_pcds/orig_coord_system/IFL_phantom_reverse_trafo.txt")
        #filtered_input.transform(trafo_IFL)
        #filtered_result.transform(trafo_IFL)

        # save point cloud
        o3d.io.write_point_cloud(os.path.join(args.root_path_pcds,curr_name + "_" + str(inputs[i].shape[1]) + "_input.ply"), filtered_input)
        o3d.io.write_point_cloud(os.path.join(args.root_path_pcds,curr_name+ "_" + str(gts[i].shape[1]) + "_GT.ply"), filtered_gt)
        o3d.io.write_point_cloud(os.path.join(args.root_path_pcds,curr_name + "_" + str(results[i].shape[1]) + "_reconstruction.ply"), filtered_result)







