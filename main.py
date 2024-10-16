import os
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk


# 1. Load CSV data into a NumPy array
csv_path = 'cts/overview.csv'

csv_headers = ["ID","Age","Contrast","ContrastTag","raw_input_path","id","tiff_name","dicom_name"]

def main():
    # Load CSV data into NumPy array
    csv_data = load_csv_to_numpy(csv_path)
    # print("CSV data loaded into NumPy array:", csv_data)
    row_num = 0
    for row in csv_data:
        if(row_num % 10 == 0):
            print('\t'.join(map(str, csv_headers)))
        print('\t'.join(map(str, row)))
        row_num += 1

    # Directory paths
    dicom_dir = 'cts/dicom_dir/'  # Path where your DICOM files are stored
    output_dir = 'simulated_drrs/'  # Output directory for DRRs

    # Generate DRRs for all DICOM files
    process_dicom_files(dicom_dir, output_dir, sim_nums=5)

    # More stuff

def load_csv_to_numpy(csv_path):
    # Load CSV as a pandas dataframe first
    df = pd.read_csv(csv_path)
    
    # Convert dataframe to a NumPy array
    data_np = df.to_numpy()
    
    return data_np

# 2. Generate DRR X-rays for each DICOM file
def generate_drr(ct_image, output_dir, sim_nums=5):
    """
    Generates DRR images from a 3D CT volume (DICOM) at varying projection angles.
    
    Parameters:
    - ct_image (SimpleITK Image): The 3D CT volume.
    - output_dir (str): Directory where DRRs will be saved.
    - sim_nums (int): Number of simulated DRRs to generate for each CT.
    
    Returns:
    None
    """
    # DRR generation parameters
    source_to_isocenter_distance = 1000  # in mm
    isocenter_to_detector_distance = 500  # in mm
    detector_resolution = (512, 512)  # Output resolution
    
    # Rotate and generate DRRs at different angles
    for i in range(sim_nums):
        # Set the rotation angle for each DRR (varying rotation about the z-axis)
        angle = i * (360 / sim_nums)  # Vary the angle

        # Define rotation in radians
        theta = np.deg2rad(angle)

        # Define the center of the CT image
        image_center = ct_image.TransformContinuousIndexToPhysicalPoint(np.array(ct_image.GetSize()) / 2.0)

        # Create the affine transform for rotation about the z-axis
        transform = sitk.AffineTransform(3)
        transform.SetCenter(image_center)
        transform.Rotate(2, 1, theta)  # Rotate about the z-axis (2 = z-axis, 1 = y-axis)

        # Resample the CT image after applying the rotation
        rotated_ct_image = sitk.Resample(ct_image, ct_image, transform, sitk.sitkLinear, 0.0)

        # Generate the DRR (maximum intensity projection along the z-axis)
        projection = sitk.MaximumProjection(rotated_ct_image, 2)

        # Convert SimpleITK image to a NumPy array
        drr_image = sitk.GetArrayFromImage(projection)

        # Normalize the DRR image to the range [0, 255]
        drr_image = np.interp(drr_image, (drr_image.min(), drr_image.max()), (0, 255))

        # Convert the normalized image to uint8 (8-bit format, compatible with TIFF)
        drr_image_uint8 = drr_image.astype(np.uint8)

        # Save DRR as TIFF file
        output_filename = os.path.join(output_dir, f'drr_{i}.tiff')
        sitk.WriteImage(sitk.GetImageFromArray(drr_image_uint8), output_filename)

        print(f"DRR {i+1}/{sim_nums} saved: {output_filename}")

def process_dicom_files(dicom_dir, output_dir, sim_nums=5):
    """
    Processes all DICOM files in dicom_dir and generates DRRs.
    
    Parameters:
    - dicom_dir (str): Directory containing the DICOM files.
    - output_dir (str): Directory where DRRs will be saved.
    - sim_nums (int): Number of DRRs to generate per DICOM.
    
    Returns:
    None
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each DICOM file in the directory
    for dicom_file in os.listdir(dicom_dir):
        if dicom_file.endswith('.dcm'):
            dicom_path = os.path.join(dicom_dir, dicom_file)

            # Read DICOM file using pydicom
            ds = pydicom.dcmread(dicom_path)

            # Convert DICOM to a 3D volume (assuming it's a CT)
            ct_image = sitk.ReadImage(dicom_path)

            # Output folder for DRRs for this DICOM file
            drr_output_dir = os.path.join(output_dir, dicom_file.split('.')[0])
            os.makedirs(drr_output_dir, exist_ok=True)

            # Generate DRR
            generate_drr(ct_image, drr_output_dir, sim_nums=sim_nums)

# Main function
if __name__ == "__main__":
    main()