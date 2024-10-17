# ct_to_x_ray

This is a repository containing exploration of getting DRR images from CTs and implementing a rudimentary version of [2], the first (to their knowledge) CNN-based 2D/3D registration method. Read below for overview, or read paper for detail. This repository implements a single zone, which can handle small perturbations of < 30 in each parameter. I also do not do the feature extraction step and register the whole image instead. The paper put the CAD image of a distal plate into the X-rays to help registration, so the task is different, but the idea is the same. The distal plate helps a lot with signal to register correctly though.

## Dataset

CTs come from: https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans/data
This dataset contains 20 CT scans of patients diagnosed with COVID-19 as well as segmentations of lungs and infections made by experts.

## Papers looked at 

1. 3-D/2-D registration of CT and MR to X-ray images (2003): https://pubmed.ncbi.nlm.nih.gov/14606674/
Non-deep registration approach. Not picked because method assumes that surfaces of bony structures are extracted preoperatively from CT or MR images

2. REAL-TIME 2D/3D REGISTRATION VIA CNN REGRESSION (2016): https://arxiv.org/pdf/1507.07505
First deep CNN registration method. Uses a transformation vector with 6 parameters to transform the CT before creating a DRR. Uses a hierarchical training method to focus on easier parameters first and the move on to harder ones. It also separates t_alpha and t_beta into zones, where each zone has a network that predicts. When predicting out of the zone, the multi-pass mode can transport us to the next zone over until we reach the right one. 