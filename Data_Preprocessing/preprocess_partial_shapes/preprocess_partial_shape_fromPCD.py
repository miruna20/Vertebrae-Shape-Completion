import argparse
import os.path
import open3d as o3d
import numpy as np
import re

"""
Note: please make sure that the .nii.gz files contain the substring "verLev" followed by a 
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
    o3d.io.write_point_cloud("/home/miruna20/Documents/PhD/PatientDataPreprocessing/Data/preProc_testing/real_pcd.pcd", real_pcd)

    real_pcd.transform(reg_result.transformation)

    return real_pcd,synthetic_pcd,translation,reg_result.transformation

def preprocess_partial_shape(vert_pcd_path, synth_template_path):
    """
    Apply preprocessing steps on vertebra segm from US which is already in pcd file
    :param vert_segm_path: path to the vertebrae partial pcd(*.pcd)
    :param synth_template_path: path to the synthetic point cloud used as alignment template
    """
    file_name = os.path.basename(vert_pcd_path)[:-4]

    pcd = o3d.io.read_point_cloud(vert_pcd_path)

    center = pcd.get_center()
    pcd.translate(-center)

    # scale down pcd
    scale = 0.01
    pcd.scale(scale,center=(0,0,0))

    # register to synthetic pipeline
    synthetic_pcd = o3d.io.read_point_cloud(synth_template_path)
    # This is the scaled and centered
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(vert_pcd_path), file_name + "_before_applying_ICP.pcd"), pcd)

    aligned_pcd, synthetic_pcd,transl,ICP_trafo = align_real_to_synthetic(pcd,synthetic_pcd)

    # save result in the same folder
    save_to_path = os.path.join(os.path.dirname(vert_pcd_path), file_name + "_transformed.pcd")
    #o3d.visualization.draw_geometries([aligned_pcd])
    o3d.io.write_point_cloud(save_to_path, aligned_pcd)

    # find the trafo matrix
    # matrix = [(ICP_trafo)⁻1] * matrix that scales by 100 and moves point to x,y,z
    inverse_trafo = np.linalg.inv(ICP_trafo)

    transl_from_synth = np.array([[1, 0, 0, -transl[0]],
                                  [0, 1, 0, -transl[1]],
                                  [0, 0, 1, -transl[2]],
                                  [0, 0, 0, 1]])
    scale_up =          np.array([[100, 0, 0, 0],
                                  [0, 100, 0, 0],
                                  [0, 0, 100 ,0],
                                  [0, 0, 0, 1]])
    trans_to_initial_pose =     np.array([[1, 0, 0, center[0]-aligned_pcd.get_center()[0]],
                                  [0, 1, 0, center[1]-aligned_pcd.get_center()[1]],
                                  [0, 0, 1 ,center[2]-aligned_pcd.get_center()[2]],
                                  [0, 0, 0, 1]])
    # we want to first apply the inverse trafo and then the trafo scales and reverse (so we need to multiply in the reverse order)
    combined_trafo = trans_to_initial_pose @ scale_up @ transl_from_synth @ inverse_trafo
    aligned_pcd.transform(combined_trafo)

    o3d.io.write_point_cloud(os.path.join(os.path.dirname(vert_pcd_path), file_name + "_transformedback.pcd"), aligned_pcd)

    # save the transformation matrix from centered to orig
    trafo_txt_file = os.path.join(os.path.dirname(vert_pcd_path),file_name + "_trafo_to_orig.txt")
    np.savetxt(trafo_txt_file,combined_trafo,fmt="%d")

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

    # iterate over all vertebrae segmentation paths
    with open(vert_list, 'r') as file:
        for US_vert_segm_path in file:

            US_vert_segm_path = US_vert_segm_path.replace("\n","")
            # verify the path has a .nii.gz ending
            if not US_vert_segm_path.endswith('.pcd'):
                raise Exception("Path does not end in .pcd: " + US_vert_segm_path)

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




