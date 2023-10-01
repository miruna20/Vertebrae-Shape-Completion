import argparse
import os.path
import SimpleITK as sitk
from skimage import measure
import open3d as o3d
import numpy as np
import re

"""
Note: please make sure that the .mhd files contain the substring "verLev" followed by a 
number between 20 and 24 which indicates the lumbar vertebra level: 
L1 --> "verLev20" 
L2 --> "verLev21"
L3 --> "verLev22"
L4 --> "verLev23"
L5 --> "verLev24"
"""

def align_real_to_synthetic(real_pcd,synthetic_pcd):
    """
    Align the real input point cloud with a synthetic template of the same level
    """

    # match the centers of the 2 point clouds
    center_synthetic = np.asarray(synthetic_pcd.get_center())
    center_real = np.asarray(real_pcd.get_center())
    translation = center_synthetic - center_real
    real_pcd.translate(translation)

    # run ICP on them so that we get an aligned real dataset
    # Set the ICP convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=200)

    # Perform ICP registration
    reg_result = o3d.pipelines.registration.registration_icp(real_pcd, synthetic_pcd, 0.1, np.identity(4),
                                                   o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                             criteria)
    real_pcd.transform(reg_result.transformation)

    return real_pcd,synthetic_pcd

def preprocess_partial_shape(vert_segm_path, synth_template_path):
    """
    Apply preprocessing steps on vertebra segm from US
    :param vert_segm_path: path to the vertebrae segmentation (*.mha)
    :param synth_template_path: path to the synthetic point cloud used as alignment template
    """

    # load .mha
    reader = sitk.ImageFileReader()
    reader.SetFileName(vert_segm_path)  # Give it the mha file as a string
    reader.LoadPrivateTagsOn()  # Make sure it can get all the info
    reader.ReadImageInformation()
    image = reader.Execute()
    res = reader.GetSpacing()
    array = sitk.GetArrayFromImage(image)
    array = np.transpose(array, (2, 1, 0))
    res = np.transpose(res)

    # transform volume to mesh with marching cubes
    verts, faces, normals, values = measure.marching_cubes(array)
    verts = verts * res

    # transform mesh to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # rotate to match the orientation of the segmentation
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=(0,0,0))
    pcd.translate(-pcd.get_center())

    # scale down pcd
    pcd.scale(0.01,center=(0,0,0))

    # register to synthetic pipeline
    synthetic_pcd = o3d.io.read_point_cloud(synth_template_path)
    aligned_pcd, synthetic_pcd = align_real_to_synthetic(pcd,synthetic_pcd)

    # save result in the same folder
    file_name = os.path.basename(vert_segm_path)[:-4]
    save_to_path = os.path.join(os.path.dirname(vert_segm_path), file_name + ".pcd")
    o3d.io.write_point_cloud(save_to_path, aligned_pcd)



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Preprocess 3D US segmentation")
    arg_parser.add_argument(
        "--vert_list",
        required=True,
        dest="vert_list",
        help="Txt file with paths to US vertebrae annotations"
    )

    args = arg_parser.parse_args()
    vert_list = args.vert_list

    # iterate over all partial pcd paths
    with open(vert_list, 'r') as file:
        for US_vert_segm_path in file:

            US_vert_segm_path = US_vert_segm_path.replace("\n","")
            # verify the path has a .mha ending
            if not US_vert_segm_path.endswith('.mha'):
                raise Exception("Path does not end in .mha: " + US_vert_segm_path)

            # verify that the file exists
            if not os.path.isfile(US_vert_segm_path):
                raise Exception("File does not exist " + US_vert_segm_path)

            # identify the vertebra level by substring "verLev"
            basename = os.path.basename(US_vert_segm_path)
            if "verLev" not in basename:
                raise Exception("Vert segm file name does not contain the vertebra level e.g L1: verLev20, L2: verLev21, etc")

            match = re.search(r'verLev(\d+)', basename)
            number = match.group(1)

            synth_template_path = os.path.join("synth_templates","template_verLev" + number + ".pcd")
            preprocess_partial_shape(US_vert_segm_path,synth_template_path)




