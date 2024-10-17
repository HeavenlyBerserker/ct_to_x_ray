# ct_to_x_ray

This is a repository containing exploration of getting DRR images from CTs and attempting some simple 2D/3D registration methods.

## Dataset

CTs come from: https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans/data
This dataset contains 20 CT scans of patients diagnosed with COVID-19 as well as segmentations of lungs and infections made by experts.

## Papers looked at 

1. 3-D/2-D registration of CT and MR to X-ray images: https://pubmed.ncbi.nlm.nih.gov/14606674/
2003 non-deep registration approach. Not picked because method assumes that surfaces of bony structures are extracted preoperatively from CT or MR images

2. 