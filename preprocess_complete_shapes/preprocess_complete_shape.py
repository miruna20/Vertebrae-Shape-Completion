import argparse
import utils_complete_shape
from skimage import measure
import open3d as o3d
import os
import nibabel as nib
import numpy as np


def separate_spine_into_vertebrae(path_segmented_spine_ct):
    # verify the path has a .nii.gz ending
    if not path_segmented_spine_ct.endswith('.nii.gz'):
        raise Exception("Path does not end in .nii.gz: " + path_segmented_spine_ct)

    # verify that the file exists
    if not os.path.isfile(path_segmented_spine_ct):
        raise Exception("File does not exist " + path_segmented_spine_ct)

    # read the segmentation with nibabel
    sample = nib.load(path_segmented_spine_ct)
    data = sample.get_fdata()

    # base folder and name
    basename = os.path.basename(path_segmented_spine_ct)[:-7]
    base_folder = os.path.dirname(path_segmented_spine_ct)

    lumbar_levels = [20, 21, 22, 23, 24]
    vertebrae = []
    for level in lumbar_levels:
        if (level in data):
            print("level" + str(level))

            vert_path = os.path.join(base_folder, basename + "_verLev" + str(level) + ".nii.gz")

            vert_data = np.zeros([data.shape[0], data.shape[1], data.shape[2]])

            indices = np.transpose((data == level).nonzero())

            for indic in indices:
                vert_data[indic[0], indic[1], indic[2]] = 1

            vert_image = nib.Nifti1Image(vert_data, sample.affine, sample.header)
            vertebrae.append(vert_image)

            # save the segmentation as intermediary representation
            if False:
                nib.save(vert_image, vert_path)

    sample.uncache()
    return vertebrae

def preprocess_partial_shape(path_segmented_spine_ct):
    """
    Apply preprocessing steps on vertebra segm from CT_spine_segm
    :param CT_spine_segm: path to the annotated CT volume segmentation (*.nii.gz)
    :return:
    """

    # separate into segmentations of 5 vertebrae, vertebrae is a list of nifti images
    vertebrae = separate_spine_into_vertebrae(path_segmented_spine_ct)

    # for each vert
    for index, vert in enumerate(vertebrae):
        # trafo segm to mesh, apply marching cubes
        pixel_dim = vert.header.get_zooms()
        verts, faces, normals, values = measure.marching_cubes(vert.get_fdata())
        verts = verts * pixel_dim

        # solve normals without point mirroring -> permute second and third coordinate of faces
        for face in faces:
                temp = face[1]
                face[1] = face[2]
                face[2] = temp

        # create an open3d mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

        # scale mesh down to 0.01
        vert_center = mesh.get_center()
        mesh.scale(0.01, center=vert_center)

        # move to center
        verts_vert = mesh.vertices - vert_center
        mesh.vertices = o3d.utility.Vector3dVector(verts_vert)

        base_folder = os.path.dirname(path_segmented_spine_ct)
        base_name = os.path.basename(path_segmented_spine_ct)[:-7]
        vert_path = os.path.join(base_folder, base_name + "_verLev" + str(20 + index) + ".obj")

        # save as obj in the same folder
        o3d.io.write_triangle_mesh(vert_path, mesh)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Preprocess 3D CT segmentations")
    arg_parser.add_argument(
        "--spine_list",
        required=True,
        dest="spine_list",
        help="Txt file with paths to the annotated CT volumes"
    )
    args = arg_parser.parse_args()
    CT_list = args.spine_list

    # iterate over all annotated CT volumes
    with open(CT_list, 'r') as file:
        for CT_spine_annotation_path in file:

            CT_spine_annotation_path = CT_spine_annotation_path.replace("\n", "")

            # verify the path has a .nii.gz ending
            if not CT_spine_annotation_path.endswith('.nii.gz'):
                raise Exception("Path does not end in .nii.gz: " + CT_spine_annotation_path)

            # verify that the file exists
            if not os.path.isfile(CT_spine_annotation_path):
                raise Exception("File does not exist " + CT_spine_annotation_path)

            preprocess_partial_shape(CT_spine_annotation_path)




