# Cooperation by non-kin during birth underpins sperm whale social complexity  🐳

**[📰 Paper](LinkTODO)** | **[📹 Explainer Video](LinkTODO)**

✍️ [Alaa Maalouf](https://alaamaalouf.github.io/), [Joseph DelPreto](https://www.josephdelpreto.com/), [Maxime Lucas](https://maximelucas.github.io/), [Simone Poetto](https://scholar.google.com/citations?user=xF_yUXEAAAAJ&hl=en), [Jacob Andreas](https://www.mit.edu/~jda/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Shane Gero]( https://carleton.ca/biology/people/shane-gero/), [Giovanni Petri](https://www.networkscienceinstitute.org/people/giovanni-petri),  [Daniela Rus](https://danielarus.csail.mit.edu/), and [David Gruber](https://www.davidgruber.com/) ✍️

🏛️ **[Project CETI]([https://www.csail.mit.edu/](https://www.projectceti.org/))** 

## Code for the ML pipeline 🐋
The code in `track_whales_and_calc_features.py` is for tracking and aligning whales based on provided segmentation masks and computing features such as orientation, bounding boxes, and more 🌊.

It was used to generate the data as part of the paper "Cooperation by non-kin during birth underpins sperm whale social complexity".




#### Challenges This Code Addresses
![challenges](/pipeline_illustrations/Challenges.jpg?raw=true)

#### Pieline for Tracking and Alignment:
![tracking design](/pipeline_illustrations/Track_align.jpg?raw=true)

#### Pipeline for Orientation Calculation
![Orientation design](/pipeline_illustrations/Orientations.jpg?raw=true)

#### Download
First, download and install the following repository: https://github.com/z-x-yang/Segment-and-Track-Anything

#### Files
Copy "track_whales_and_calc_features.py" and "Segmentations.py" to "Segment-and-Track-Anything" directory. 

#### Verify
Check that the file "model_args.py" points to the right paths.

#### Data 
Use your segmentation numpy file output of yolo and store them in the following format:

<segmentation_data_dir>/masks_npy

<segmentation_data_dir>/masks

The masks_npy folder should contain the raw segmentation outputs as NumPy arrays.
The masks folder should contain visualizations of these segmentations.
These visualizations are used to manually inspect the segmentation results and remove bad frames before running the tracking and alignment code, helping to ensure that poor-quality segmentations are excluded from the pipeline.

#### Example command:

python track_whales_and_calc_features.py --input_video_name 1688827660979 --input_video_path /home/data/1688827660979.mp4 --folder_to_store output_1688827660979 --seg_path /home/data/segs/1688827660979/  

#### Pre-generated Segmentation Data and Raw Videos
To download example segmentation masks and raw video for running the code, use the following link: Link Will Be Provided Soon.

#### Note:
In the paper, we ran this process interactively using the resume_from argument. This allowed us to iteratively refine the results, applying many manual corrections using our interactive tool—particularly in cases where whales went underwater or where segmentation was distorted.

## Code for the network analysis

Can be found under `network_analysis/`, divided into `scripts/` with helper functions, and `notebooks/` with the analyses used for the figures. 
