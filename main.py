import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import numpy as np
import re
import random
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm  # Import tqdm for progress bars


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Main execution
def main():
    # Network parameters
    batch_size = 80
    num_epochs = 100
    device = "cuda:0"
    train_portion = .8

    # Define file paths and directories
    metadata_csv = "cts/metadata.csv"  # Path to metadata.csv
    ct_scan_dir_og = "cts/ct_scans/"
    ct_scan_dir = "cts/ct_scans_resampled/"  # Directory containing .nii CT scans
    drr_base_dir = "simulated_drrs/"  # Directory where simulated DRRs will be saved

    # # Step 1: Find the maximum shape that can fit all CT scans
    # max_shape = find_max_shape(ct_scan_dir_og)
    # print(f"Max shape: {max_shape}")
    
    # # Step 2: Pad all CT scans to the max shape and save them
    # pad_and_save_ct_scans(ct_scan_dir_og, max_shape)

    # # Downsampling ct scans to a new directory
    # process_downsample_ct_scans(ct_scan_dir_og, ct_scan_dir, downsample_factor=4)

    # # Process CT scans and generate DRRs
    # process_ct_scans(metadata_csv, ct_scan_dir, drr_base_dir, num_samps=500)

    # List all CT file paths
    ct_fns = list_ct_file_paths(ct_scan_dir)

    # Dict of all DRR file paths and parameters generated from each ct_fns
    dict_drr = get_dict_drr_paths(drr_base_dir, ct_fns)

    # Check the results
    # for ct_fn, drr_paths in dict_drr_paths.items():
    #     print(f"CT File: {ct_fn}")
    #     print("DRR Files:", drr_paths)

    train_ct_fns, test_ct_fns, train_dict_drr, test_dict_drr = split_train_test(ct_fns, dict_drr, train_ratio=train_portion)

    # for ct_fn, drr_paths in test_dict_drr.items():
    #     print(f"CT File: {ct_fn}")
    #     print("DRR Files:", drr_paths)

    '''Setup network'''

    # Create PyTorch datasets for the training and test sets
    train_dataset = DRRDataset(train_dict_drr, device=device)
    test_dataset = DRRDataset(test_dict_drr, device=device)

    # Example: Create data loaders (optional, but common in PyTorch workflows)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the three networks and move them to GPU
    reg_net_g1 = BaseRegNet(output_size=3).to(device)  # Outputs 3 values
    reg_net_g2 = BaseRegNet(output_size=2).to(device)  # Outputs 2 values
    reg_net_g3 = BaseRegNet(output_size=1).to(device)  # Outputs 1 value

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer for all three networks together and move to GPU
    optimizer = optim.Adam(
        list(reg_net_g1.parameters()) + 
        list(reg_net_g2.parameters()) + 
        list(reg_net_g3.parameters()), 
        lr=0.001
    )

    # Main training loop with correct MSE and MAD reporting for parameters
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        reg_net_g1.train()  # Set networks to training mode
        reg_net_g2.train()
        reg_net_g3.train()

        epoch_loss = 0.0  # Track the total MSE loss for the epoch
        t_x_loss, t_y_loss, t_z_loss = 0.0, 0.0, 0.0  # For g1 outputs
        t_theta_loss, t_alpha_loss = 0.0, 0.0  # For g2 outputs
        t_beta_loss = 0.0  # For g3 output
        num_samples = 0  # To track number of samples

        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            for ct, data, target in tepoch:
                batch_size = data.size(0)
                num_samples += batch_size

                optimizer.zero_grad()  # Zero the gradients

                # Initial batch-size zeros for transformations
                curr_t = torch.zeros((data.size(0), 6)).to(device)

                # Forward pass for first network
                output1 = reg_net_g1(data)
                curr_t[:, :3] = output1
                drr_1 = generate_batched_drr_with_transformations(ct, curr_t)

                # Forward pass for second network
                output2 = reg_net_g2(drr_1 - data)
                curr_t[:, 3:5] = output2
                drr_2 = generate_batched_drr_with_transformations(ct, curr_t)

                # Forward pass for third network
                output3 = reg_net_g3(drr_2 - data)
                curr_t[:, 5:6] = output3

                # Compute the total loss between predicted transformations and target (MSE)
                total_loss = criterion(curr_t, target)

                # Compute mean absolute differences for each transformation parameter for reporting only
                t_x_loss += torch.sum(torch.abs(curr_t[:, 0] - target[:, 0])).item()
                t_y_loss += torch.sum(torch.abs(curr_t[:, 1] - target[:, 1])).item()
                t_z_loss += torch.sum(torch.abs(curr_t[:, 2] - target[:, 2])).item()
                t_theta_loss += torch.sum(torch.abs(curr_t[:, 3] - target[:, 3])).item()
                t_alpha_loss += torch.sum(torch.abs(curr_t[:, 4] - target[:, 4])).item()
                t_beta_loss += torch.sum(torch.abs(curr_t[:, 5] - target[:, 5])).item()

                # Backward pass (compute gradients)
                total_loss.backward()

                # Update weights for all three networks
                optimizer.step()

                # Accumulate the total MSE loss for the epoch (MSE is already averaged per batch)
                epoch_loss += total_loss.item()

                # Update the progress bar with individual parameter MADs per sample
                tepoch.set_postfix(
                    total_loss=epoch_loss / len(train_loader),  # MSE loss averaged over all batches
                    t_x=t_x_loss / num_samples,  # MAD for t_x
                    t_y=t_y_loss / num_samples,  # MAD for t_y
                    t_z=t_z_loss / num_samples,  # MAD for t_z
                    t_θ=t_theta_loss / num_samples,  # MAD for θ (theta)
                    t_α=t_alpha_loss / num_samples,  # MAD for α (alpha)
                    t_β=t_beta_loss / num_samples  # MAD for β (beta)
                )

        print(f"Epoch {epoch+1} completed. Average train loss (MSE): {epoch_loss/len(train_loader):.4f}")

        # Testing at the end of each epoch
        avg_test_loss = test_model(test_loader, reg_net_g1, reg_net_g2, reg_net_g3, criterion, device)

# Testing function with reporting parameter-wise mean absolute differences (MAD)
def test_model(test_loader, reg_net_g1, reg_net_g2, reg_net_g3, criterion, device):
    reg_net_g1.eval()  # Set networks to evaluation mode
    reg_net_g2.eval()  
    reg_net_g3.eval()  

    test_loss = 0.0  # Track the total test loss (MSE)
    t_x_loss, t_y_loss, t_z_loss = 0.0, 0.0, 0.0  # For g1 outputs
    t_theta_loss, t_alpha_loss = 0.0, 0.0  # For g2 outputs
    t_beta_loss = 0.0  # For g3 output
    num_samples = 0  # Track the number of samples

    with torch.no_grad():  # No gradient computation in the test phase
        with tqdm(test_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Testing")
            for ct, data, target in tepoch:
                batch_size = data.size(0)
                num_samples += batch_size

                # Move data to the GPU
                ct = ct.to(device)
                data = data.to(device)
                target = target.to(device)

                # Initial batch-size zeros for transformations
                curr_t = torch.zeros((data.size(0), 6)).to(device)

                # Forward pass for first network
                output1 = reg_net_g1(data)
                curr_t[:, :3] = output1
                drr_1 = generate_batched_drr_with_transformations(ct, curr_t)

                # Forward pass for second network
                output2 = reg_net_g2(drr_1 - data)
                curr_t[:, 3:5] = output2
                drr_2 = generate_batched_drr_with_transformations(ct, curr_t)

                # Forward pass for third network
                output3 = reg_net_g3(drr_2 - data)
                curr_t[:, 5:6] = output3

                # Compute the total loss (MSE)
                total_loss = criterion(curr_t, target)

                # Compute mean absolute differences for each transformation parameter for reporting only
                t_x_loss += torch.sum(torch.abs(curr_t[:, 0] - target[:, 0])).item()
                t_y_loss += torch.sum(torch.abs(curr_t[:, 1] - target[:, 1])).item()
                t_z_loss += torch.sum(torch.abs(curr_t[:, 2] - target[:, 2])).item()
                t_theta_loss += torch.sum(torch.abs(curr_t[:, 3] - target[:, 3])).item()
                t_alpha_loss += torch.sum(torch.abs(curr_t[:, 4] - target[:, 4])).item()
                t_beta_loss += torch.sum(torch.abs(curr_t[:, 5] - target[:, 5])).item()

                # Accumulate the total test loss (MSE is already averaged per batch)
                test_loss += total_loss.item()

                # Update the progress bar with individual parameter MADs per sample
                tepoch.set_postfix(
                    test_loss=test_loss / len(test_loader),  # MSE loss (MSE averaged over all batches)
                    t_x=t_x_loss / num_samples,  # MAD for t_x
                    t_y=t_y_loss / num_samples,  # MAD for t_y
                    t_z=t_z_loss / num_samples,  # MAD for t_z
                    t_θ=t_theta_loss / num_samples,  # MAD for θ (theta)
                    t_α=t_alpha_loss / num_samples,  # MAD for α (alpha)
                    t_β=t_beta_loss / num_samples  # MAD for β (beta)
                )

    avg_test_loss = test_loss / len(test_loader)
    print(f"Testing completed. Average test loss (MSE): {avg_test_loss:.4f}")
    return avg_test_loss

def generate_batched_drr_with_transformations(ct_batch, param_batch):
    """
    Generates DRR images from a batch of CT volumes with batched random transformations applied.
    
    Parameters:
    - ct_batch (torch.Tensor): Batch of 3D CT volumes (B, D, H, W) where B is the batch size.
    - param_batch (torch.Tensor): Batch of transformation parameters (B, 6), including [t_x, t_y, t_z, t_theta, t_alpha, t_beta].
    
    Returns:
    - drr_batch (torch.Tensor): Batch of DRR images (B, H, W) after applying transformations and MIP.
    """
    
    # Extract transformation parameters from param_batch
    t_x, t_y, t_z = param_batch[:, 0], param_batch[:, 1], param_batch[:, 2]  # Translations
    t_theta, t_alpha, t_beta = param_batch[:, 3], param_batch[:, 4], param_batch[:, 5]  # Rotations (roll, pitch, yaw)

    # Create the affine transformation matrices for each transformation
    B, D, H, W = ct_batch.shape  # Adjusted shape
    affine_matrices = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(ct_batch.device)  # (B, 3, 3)

    # Convert angles to radians
    t_theta_rad = torch.deg2rad(t_theta)
    t_alpha_rad = torch.deg2rad(t_alpha)
    t_beta_rad = torch.deg2rad(t_beta)

    # Apply translation
    translation_matrix = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(ct_batch.device)
    translation_matrix[:, :3, 3] = torch.stack([t_x, t_y, t_z], dim=-1)  # (B, 3)

    # Apply roll (rotation around the x-axis)
    roll_matrix = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(ct_batch.device)
    roll_matrix[:, 1, 1] = torch.cos(t_theta_rad)
    roll_matrix[:, 1, 2] = -torch.sin(t_theta_rad)
    roll_matrix[:, 2, 1] = torch.sin(t_theta_rad)
    roll_matrix[:, 2, 2] = torch.cos(t_theta_rad)

    # Apply pitch (rotation around the y-axis)
    pitch_matrix = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(ct_batch.device)
    pitch_matrix[:, 0, 0] = torch.cos(t_alpha_rad)
    pitch_matrix[:, 0, 2] = torch.sin(t_alpha_rad)
    pitch_matrix[:, 2, 0] = -torch.sin(t_alpha_rad)
    pitch_matrix[:, 2, 2] = torch.cos(t_alpha_rad)

    # Apply yaw (rotation around the z-axis)
    yaw_matrix = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(ct_batch.device)
    yaw_matrix[:, 0, 0] = torch.cos(t_beta_rad)
    yaw_matrix[:, 0, 1] = -torch.sin(t_beta_rad)
    yaw_matrix[:, 1, 0] = torch.sin(t_beta_rad)
    yaw_matrix[:, 1, 1] = torch.cos(t_beta_rad)

    # Combine transformations
    full_transform = yaw_matrix @ pitch_matrix @ roll_matrix @ translation_matrix

    # Add the channel dimension to ct_batch for grid_sample, making it (B, 1, D, H, W)
    ct_batch = ct_batch.unsqueeze(1)  # Adding a single channel dimension

    # Apply transformations using grid_sample (affine_grid and grid_sample)
    grid = F.affine_grid(full_transform[:, :3, :], ct_batch.size(), align_corners=False)
    ct_batch = ct_batch.float()
    transformed_ct_batch = F.grid_sample(ct_batch, grid, mode='bilinear', align_corners=False)

    # Compute maximum intensity projection (MIP) along the z-axis for DRR generation
    drr_batch = torch.max(transformed_ct_batch, dim=2)[0]  # Maximum along the z-axis (depth dimension)

    # First, take the min over dimension 2 (height)
    drr_batch_min = drr_batch.min(dim=2, keepdim=True)[0]  # Min along the height (axis 2)

    # Now, take the min over dimension 3 (width)
    drr_batch_min = drr_batch_min.min(dim=3, keepdim=True)[0]  # Min along the width (axis 3)

    # Similarly for max
    drr_batch_max = drr_batch.max(dim=2, keepdim=True)[0]  # Max along height (axis 2)
    drr_batch_max = drr_batch_max.max(dim=3, keepdim=True)[0]  # Max along width (axis 3)

    # Normalize the DRR batch
    drr_batch = (drr_batch - drr_batch_min) / (drr_batch_max - drr_batch_min + 1e-6)  # Add epsilon to avoid division by zero
    
    return drr_batch  # (B, H, W)

class BaseRegNet(nn.Module):
    def __init__(self, output_size):
        super(BaseRegNet, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 2 poolings: 158/(2^2) = 39 (height and width)
        self.fc1 = nn.Linear(32 * 39 * 39, 250)  # Adjust for 2 conv + pool layers
        self.fc2 = nn.Linear(250, output_size)  # Output size varies for each network

    def forward(self, x):
        # Forward through the convolutional layers + pooling
        x = self.pool(F.relu(self.conv1(x)))  # 158 -> 79
        x = self.pool(F.relu(self.conv2(x)))  # 79 -> 39
        
        # Flatten the tensor
        x = x.view(-1, 32 * 39 * 39)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # # Apply tanh activation
        # tanhx = torch.tanh(x)

        # # Create a new tensor for scaling without in-place operations
        # scaledx = torch.zeros_like(tanhx)
        # scaledx[:, :3] = tanhx[:, :3] * 158  # Scale the first 3 outputs by 158
        # scaledx[:, 3:] = tanhx[:, 3:] * 180  # Scale the rest of the outputs by 180

        return x

class DRRDataset(Dataset):
    """
    PyTorch Dataset for DRRs and their corresponding transformation parameters.
    """
    def __init__(self, drr_dict, device):
        """
        Initialize the dataset.
        
        Parameters:
        - drr_dict (dict): A dictionary where keys are CT file paths, and values are lists of [DRR path, t_x, t_y, t_z, t_alpha, t_beta, t_epsilons].
        - device (torch.device): The device (CPU or GPU) where data should be loaded.
        """
        self.drr_dict = drr_dict
        self.device = device
        
        # Create a list of (drr_path, [t_x, t_y, t_z, t_alpha, t_beta, t_epsilons])
        self.data = []
        self.ct_fn = []
        self.ct = {}  # Dictionary to store CT images in memory
        self.drr_images = {}  # Dictionary to store DRR images in memory
        
        for ct_fn, drr_list in drr_dict.items():
            # Load and store the CT image in GPU memory
            ct_image = sitk.ReadImage(ct_fn)
            np_image = sitk.GetArrayFromImage(ct_image)
            torch_image = torch.from_numpy(np_image).float().to(self.device)  # Move to GPU
            self.ct[ct_fn] = torch_image
            
            for drr_info in drr_list:
                drr_path = drr_info[0]
                # Load and store DRR images in GPU memory
                drr_image = sitk.ReadImage(drr_path)
                drr_image_np = sitk.GetArrayFromImage(drr_image)  # Convert to NumPy array
                drr_image_tensor = torch.tensor(drr_image_np, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add channel dimension and move to GPU
                self.drr_images[drr_path] = drr_image_tensor  # Store in the dictionary
                
                # Append the DRR path and corresponding transformation parameters
                self.data.append(drr_info)  # (drr_path, t_x, t_y, t_z, t_alpha, t_beta, t_epsilons)
                self.ct_fn.append(ct_fn)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the DRR image and its corresponding transformation parameters.
        
        Parameters:
        - idx (int): Index of the item to get.
        
        Returns:
        - ct_image (Tensor): The corresponding CT image as a tensor.
        - image_tensor (Tensor): The DRR image as a tensor.
        - transform_params (Tensor): The transformation parameters as a tensor.
        """
        drr_info = self.data[idx]
        drr_path = drr_info[0]
        transform_params = torch.tensor(drr_info[1:], dtype=torch.float32).to(self.device)  # Convert to a tensor and move to GPU
        
        # Fetch the DRR image from GPU memory
        image_tensor = self.drr_images[drr_path]
        
        # Fetch the corresponding CT image from GPU memory
        ct_image = self.ct[self.ct_fn[idx]]
        
        return ct_image, image_tensor, transform_params


def split_train_test(ct_fns, dict_drr, train_ratio=0.8):
    """
    Split the CT file paths and corresponding DRR dictionary into 80% train and 20% test.

    Parameters:
    - ct_fns (list): List of CT file paths.
    - dict_drr (dict): Dictionary where keys are CT file paths and values are lists of DRR file paths and parameters.
    - train_ratio (float): Ratio of data for the training set (default is 0.8 for 80%).

    Returns:
    - train_ct_fns (list): CT file paths for the training set.
    - test_ct_fns (list): CT file paths for the test set.
    - train_dict_drr (dict): DRR dictionary for the training set.
    - test_dict_drr (dict): DRR dictionary for the test set.
    """
    
    # Shuffle the CT file paths
    random.shuffle(ct_fns)
    
    # Compute the split index
    split_idx = int(train_ratio * len(ct_fns))
    
    # Split the CT file paths into train and test
    train_ct_fns = ct_fns[:split_idx]
    test_ct_fns = ct_fns[split_idx:]
    
    # Create train and test DRR dictionaries based on the split CT paths
    train_dict_drr = {ct_fn: dict_drr[ct_fn] for ct_fn in train_ct_fns}
    test_dict_drr = {ct_fn: dict_drr[ct_fn] for ct_fn in test_ct_fns}
    
    return train_ct_fns, test_ct_fns, train_dict_drr, test_dict_drr

def list_ct_file_paths(ct_scan_dir):
    """
    Lists all CT file paths in the given directory.

    Parameters:
    - ct_scan_dir (str): Directory containing CT scan files.

    Returns:
    - ct_fns (list): List of CT file paths.
    """
    ct_fns = []
    for root, dirs, files in os.walk(ct_scan_dir):
        for file in files:
            if file.endswith('.nii'):  # Assuming the CT files have a .nii extension
                ct_fns.append(os.path.join(root, file))
    return ct_fns


def parse_drr_filename(filename):
    """
    Parse the DRR filename to extract the transformation parameters.

    Expected filename format: drr_tx{t_x}_ty{t_y}_tz{t_z}_r{t_alpha}_p{t_beta}_y{t_epsilons}.tiff

    Parameters:
    - filename (str): The DRR file name.

    Returns:
    - List of extracted parameters: [t_x, t_y, t_z, t_alpha, t_beta, t_epsilons].
    """
    # Extract the values using regular expressions
    pattern = r'drr_tx(?P<t_x>-?\d+\.\d+)_ty(?P<t_y>-?\d+\.\d+)_tz(?P<t_z>-?\d+\.\d+)_r(?P<t_alpha>-?\d+\.\d+)_p(?P<t_beta>-?\d+\.\d+)_y(?P<t_epsilons>-?\d+\.\d+)'
    match = re.search(pattern, filename)

    if match:
        try:
            t_x = float(match.group('t_x'))
            t_y = float(match.group('t_y'))
            t_z = float(match.group('t_z'))
            t_alpha = float(match.group('t_alpha'))
            t_beta = float(match.group('t_beta'))
            t_epsilons = float(match.group('t_epsilons'))
            
            return [t_x, t_y, t_z, t_alpha, t_beta, t_epsilons]
        except ValueError:
            print(f"Error parsing parameters from filename: {filename}")
            return [None] * 6  # Return list with None values if parsing fails
    else:
        print(f"Unexpected filename format: {filename}")
        return [None] * 6

def get_dict_drr_paths(drr_base_dir, ct_fns):
    """
    Creates a dictionary where keys are CT file paths and values are lists of [DRR file path, t_x, t_y, t_z, t_alpha, t_beta, t_epsilons].

    Parameters:
    - drr_base_dir (str): Base directory where DRR directories are stored.
    - ct_fns (list): List of CT file paths.

    Returns:
    - drr_dict (dict): Dictionary where keys are CT file paths and values are lists of DRR file path and transformation parameters.
    """
    drr_dict = {}

    for ct_fn in ct_fns:
        ct_name = os.path.splitext(os.path.basename(ct_fn))[0]
        drr_dir = os.path.join(drr_base_dir, ct_name)
        
        if os.path.exists(drr_dir):
            # List all DRR files in the directory
            drr_fns = sorted([os.path.join(drr_dir, f) for f in os.listdir(drr_dir) if f.endswith('.tiff')])
            
            # Parse each DRR filename to extract transformation parameters
            drr_info = []
            for drr_fn in drr_fns:
                drr_filename = os.path.basename(drr_fn)
                params = parse_drr_filename(drr_filename)
                drr_info.append([drr_fn] + params)
            
            drr_dict[ct_fn] = drr_info
        else:
            print(f"DRR directory {drr_dir} not found")
            drr_dict[ct_fn] = []  # Empty list if DRR directory is missing

    return drr_dict

# Function to generate DRRs from a CT scan with added transformations
def generate_drr_with_transformations(ct_image, output_dir, num_samps=5):
    """
    Generates DRR images from a 3D CT volume with random transformations applied.

    Parameters:
    - ct_image (SimpleITK Image): The 3D CT volume.
    - output_dir (str): Directory where DRRs will be saved.
    - num_samps (int): Number of simulated DRRs to generate for each CT.
    """
    # Define the reasonable range for translations and rotations (in mm and degrees)
    t_x_range = (-10, 10)  # Translation range for x-axis
    t_y_range = (-10, 10)  # Translation range for y-axis
    t_z_range = (-20, 20)  # Translation range for z-axis (with scaling)
    t_theta_range = (-30, 30)  # Roll (in degrees)
    t_alpha_range = (-30, 30)  # Pitch (in degrees)
    t_beta_range = (-30, 30)   # Yaw (in degrees)
    t_epsilons_range = (-30, 30)  # Added random perturbation (in degrees)
    
    for i in range(num_samps):
        # Randomly sample translation and rotation parameters
        t_x = np.random.uniform(*t_x_range)
        t_y = np.random.uniform(*t_y_range)
        t_z = np.random.uniform(*t_z_range)
        t_theta = np.random.uniform(*t_theta_range)
        t_alpha = np.random.uniform(*t_alpha_range)
        t_beta = np.random.uniform(*t_beta_range)
        
        # Apply affine transformation with translation and rotation
        transform = sitk.AffineTransform(3)
        transform.Translate([t_x, t_y, t_z])

        # Convert angles from degrees to radians
        theta_rad = np.deg2rad(t_theta)  # Roll
        alpha_rad = np.deg2rad(t_alpha)  # Pitch
        beta_rad = np.deg2rad(t_beta)    # Yaw

        # Apply rotations (roll, pitch, yaw)
        transform.Rotate(2, 1, beta_rad)   # Yaw (around z-axis)
        transform.Rotate(1, 0, alpha_rad)  # Pitch (around y-axis)
        transform.Rotate(0, 2, theta_rad)  # Roll (around x-axis)

        # Set the center of rotation to the center of the image
        image_center = ct_image.TransformContinuousIndexToPhysicalPoint(np.array(ct_image.GetSize()) / 2.0)
        transform.SetCenter(image_center)
        
        # Resample the image with the applied transformation
        rotated_ct_image = sitk.Resample(ct_image, ct_image, transform, sitk.sitkLinear, 0.0)

        # Generate DRR using maximum intensity projection along the z-axis
        projection = sitk.MaximumProjection(rotated_ct_image, 2)

        # Normalize the DRR image
        drr_image = sitk.GetArrayFromImage(projection)
        drr_image = np.interp(drr_image, (drr_image.min(), drr_image.max()), (0, 255)).astype(np.uint8)

        # Save the DRR image with the transformation parameters in the filename
        output_filename = os.path.join(output_dir, f'drr_tx{t_x:.2f}_ty{t_y:.2f}_tz{t_z:.2f}_r{t_theta:.2f}_p{t_alpha:.2f}_y{t_beta:.2f}.tiff')
        sitk.WriteImage(sitk.GetImageFromArray(drr_image), output_filename)
        print(f"DRR {i+1}/{num_samps} saved: {output_filename}")

# Main function to read the CSV and generate DRRs for each CT scan
def process_ct_scans(metadata_csv, ct_scan_dir, output_base_dir, num_samps=5):
    """
    Reads metadata CSV and generates DRRs for each CT scan.
    
    Parameters:
    - metadata_csv (str): Path to the metadata.csv file.
    - ct_scan_dir (str): Directory containing .nii CT scans.
    - output_base_dir (str): Base directory where DRRs will be saved.
    - sim_nums (int): Number of DRRs to generate for each CT scan.
    
    Returns:
    None
    """
    # Load the metadata CSV
    metadata = pd.read_csv(metadata_csv).to_numpy()

    # Loop through each entry in the metadata and process the corresponding CT scan
    for entry in metadata:
        ct_filename = os.path.basename(entry[0])  # Only keep the file name, not the path
        ct_path = os.path.join(ct_scan_dir, ct_filename)  # Ensure the correct directory is used
        
        # Load the CT scan using SimpleITK
        if os.path.exists(ct_path):
            ct_image = sitk.ReadImage(ct_path)
            print(ct_image.GetSize())

            # Define the output directory for the DRRs
            output_dir = os.path.join(output_base_dir, os.path.splitext(ct_filename)[0])
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate DRRs
            generate_drr_with_transformations(ct_image, output_dir, num_samps=num_samps)
        else:
            print(f"CT scan {ct_filename} not found at {ct_path}")

def find_max_shape(ct_scan_dir):
    """
    Find the maximum shape of all 3D CT scan volumes in the directory.
    
    Parameters:
    - ct_scan_dir (str): Directory containing the .nii CT scans.
    
    Returns:
    - max_shape (tuple): The shape that can accommodate all CT scans.
    """
    max_shape = (0, 0, 0)
    
    for ct_file in os.listdir(ct_scan_dir):
        if ct_file.endswith('.nii'):
            ct_path = os.path.join(ct_scan_dir, ct_file)
            ct_img = nib.load(ct_path)
            ct_data = ct_img.get_fdata()
            max_shape = np.maximum(max_shape, ct_data.shape)
    
    return tuple(max_shape)

def pad_and_save_ct_scans(ct_scan_dir, max_shape):
    """
    Pad all CT scans to the same size and save them back to the original files.
    
    Parameters:
    - ct_scan_dir (str): Directory containing the .nii CT scans.
    - max_shape (tuple): The shape to which all volumes will be padded.
    """
    for ct_file in os.listdir(ct_scan_dir):
        if ct_file.endswith('.nii'):
            ct_path = os.path.join(ct_scan_dir, ct_file)
            ct_img = nib.load(ct_path)
            ct_data = ct_img.get_fdata()
            original_shape = ct_data.shape
            
            # Calculate padding for each axis to center the image
            padding = [(int(np.floor((m - o) / 2)), int(np.ceil((m - o) / 2))) for m, o in zip(max_shape, original_shape)]
            
            # Pad the image
            padded_ct_data = np.pad(ct_data, padding, mode='constant', constant_values=0)
            
            # Create a new NIfTI image
            padded_ct_img = nib.Nifti1Image(padded_ct_data, ct_img.affine, ct_img.header)
            
            # Save the padded image back to the same path
            nib.save(padded_ct_img, ct_path)
            print(f"Saved padded CT scan: {ct_file}")

def downsample_ct_scan(ct_scan, downsample_factor):
    """
    Downsample a 3D CT scan by the given factor.
    
    Parameters:
    - ct_scan (numpy.ndarray): The original 3D CT scan.
    - downsample_factor (float): The factor by which to downsample the scan.
    
    Returns:
    - downsampled_scan (numpy.ndarray): The downsampled 3D CT scan.
    """
    # Use scipy's zoom function to downsample the scan
    downsampled_scan = zoom(ct_scan, (1/downsample_factor, 1/downsample_factor, 1/downsample_factor), order=3)  # order=3 for cubic interpolation
    return downsampled_scan

def process_downsample_ct_scans(ct_scan_dir_og, ct_scan_dir, downsample_factor=4):
    """
    Downsample all CT scans in the original directory and save them to a new directory.
    
    Parameters:
    - ct_scan_dir_og (str): Directory containing the original CT scans.
    - ct_scan_dir (str): Directory to save the downsampled CT scans.
    - downsample_factor (float): Factor by which to downsample the scans.
    """
    # Ensure the output directory exists
    os.makedirs(ct_scan_dir, exist_ok=True)

    # Iterate over all files in the original CT scan directory
    for filename in os.listdir(ct_scan_dir_og):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            # Load the CT scan
            ct_scan_path = os.path.join(ct_scan_dir_og, filename)
            ct_image = nib.load(ct_scan_path)
            ct_data = ct_image.get_fdata()

            # Downsample the CT scan
            downsampled_data = downsample_ct_scan(ct_data, downsample_factor)

            # Create a new nibabel image with the downsampled data and the same affine
            downsampled_image = nib.Nifti1Image(downsampled_data, ct_image.affine)

            # Save the downsampled image to the new directory
            downsampled_filename = os.path.join(ct_scan_dir, filename)
            nib.save(downsampled_image, downsampled_filename)

            print(f"Downsampled {filename} and saved to {downsampled_filename}")

# Main function
if __name__ == "__main__":
    main()