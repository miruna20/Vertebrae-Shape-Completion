# Vertebrae-Shape-Completion


## Pipeline for data preprocessing 

### A) Preprocessing of US data - Partial Shape (preprocess_partial_shape.py)

Given: 3D Segmentation of the spine in US volume, all vertebrae are annotated with the same label 
1) For each spine segmentation: 
    Manually separate the vertebrae levels 
    Input: 3D segmentation of the spine 
    Output: n, n<=5 vertebrae segmentations in .mha format for each spine 
2) For each vertebra segmentation 
  
    2.1) Transform segmentation to point cloud 

    2.2) Scale down by 0.01

    2.3) Center

    2.4) Register to synthetic template   
Output: partial point cloud of vertebra
    
### B) Preprocessing of CT data - Complete Shape (preprocess_complete_shape.py) 

Given: 3D segmentation of the spine in a CT volume with individual vertebrae labels in .nii.gz format 
(E.g as a result of the Payer model) 
1) For each CT volume:
    Separate whole spine segmentation into vert segmentation based on labels
    Input: Annotated 3D CT volume of the spine 
    Output: n, n<=5 lumbar vertebrae segmentations in .nii.gz format  
2) For each vertebra segmentation: 

    2.1) Trafo vert to mesh 

    2.2) Scale down by 0.01

    2.3) Center based on the centroid of the vertebra 

    Output: Complete mesh of vertebra
    
### C) Create h5 dataset of pairs (create_dataset.py)

1) For each vertebra: 

    1.1) Apply FPS on partial pcd to obtain a point cloud with 4096 points

    1.2) Apply PDS on complete mesh to obtain a point cloud with 4096 points  
