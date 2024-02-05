# Vertebrae-Shape-Completion


## Pipeline for data preprocessing
    
### A) Preprocessing of CT data - Complete Shape (preprocess_complete_shape.py) 

Given: 3D segmentation of the spine in a CT volume with individual vertebrae labels in .nii.gz format 
(E.g as a result of the Payer model) 
1) For each CT volume:
    Separate whole spine segmentation into vert segmentation based on labels
    Input: Annotated 3D CT volume of the spine 
    Output: n, n<=5 lumbar vertebrae segmentations in .nii.gz format  
2) For each vertebra segmentation: 

    2.1) Trafo vert to mesh

    Output: Complete mesh of vertebra
    
### B) Create h5 dataset of pairs (create_dataset.py)

Given: Registered partial vertebra (pcd file) with complete vertebra shape (obj file)

1) For each vertebra: 
    1.1) Load one pcd file containing partial point cloud, load one obj file containing complete vertebra
   (This script assumes that the pcd and the obj file are aligned)

    1.2) Center, scale to unit sphere and align to synthetic template both the partial pcd and the complete vertebra 

    1.3) Apply FPS on partial pcd to obtain a point cloud with 4096 points

    1.4) Apply PDS on complete mesh to obtain a point cloud with 4096 points  
