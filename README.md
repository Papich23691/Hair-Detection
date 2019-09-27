# Hair-Detection
Hair Mask RCNN using matterport model

### Download and extract dataset and weights directly to repository folder: </br>
[Dataset](https://drive.google.com/file/d/1C-0foSYsKBh1bxp9XRIMXKUO6er4OqZc/view?usp=sharing)</br>
[Weights](https://drive.google.com/file/d/1ZbWTqWLi7w-lVvf7TQ59Gqil_SJnofbE/view?usp=sharing)</br></br>

#### Folder path should look like
.
|
├── dataset                     # Dataset
│   ├── train                   # Train images 
│   └── val                     # Validation images    
|
├── mask_rcnn_hair_0200.h5      # Weights
|
└── run.py                      # Training,loading and detecting functions
