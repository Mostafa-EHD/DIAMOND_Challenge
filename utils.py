from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch
from tqdm import tqdm 
import numpy as np
import os
import pandas as pd

def read_image(path, resize_size=None, gray=False):
    """
    Reads an image from a specified path, optionally converts it to grayscale, and resizes it.

    Args:
        path (str): The file path to the image that needs to be read.
        resize_size (tuple of int, optional): A tuple specifying the desired resize dimensions (width, height).
                                               If None, the image is not resized. Default is None.
        gray (bool): If True, converts the image to grayscale. Default is False.

    Returns:
        PIL.Image.Image: The loaded (and possibly processed) image.
    """
    img = Image.open(path)
    if gray:
        img = ImageOps.grayscale(img)
    if resize_size is not None:
        img = img.resize(resize_size)
    return img

def augment_image(PIL_img):
    """
    Applies data augmentation transformations to a PIL image.

    This function currently applies random horizontal and vertical flips to the image.
    Additional transformations can be added to the `transforms.Compose` list as needed.

    Args:
        PIL_img (PIL.Image.Image): The image to augment, in PIL format.

    Returns:
        PIL.Image.Image: The augmented image.
    """
    im_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    img_augmented = im_aug(PIL_img)
    return img_augmented



def prepare_data(data, device):
    """
    Prepares the data for processing by moving it to the specified device (CPU or GPU).

    Parameters:
    - data (dict): A dictionary containing at least 'image' and 'label' keys with their corresponding tensors.
    - device (torch.device): The device to which the data should be moved.

    Returns:
    - Tuple of tensors: The image and label tensors moved to the specified device.
    """
    img = data['image'].to(device).float()
    labels = data['label'].type(torch.LongTensor).to(device)
    return img, labels


def generate_val_csv(model, val_dataloader, device, results_dir):
    """
    GENERATES A CSV FILE CONTAINING THE PREDICTIONS FOR A VALIDATION DATASET ALONG WITH THEIR ASSOCIATED PROBABILITIES.
    
    THIS FUNCTION IS CRITICAL FOR EVALUATING THE MODEL'S PERFORMANCE ON THE VALIDATION SET. PARTICIPANTS SHOULD NOT
    CHANGE THIS FUNCTION IN ANY WAY TO ENSURE THE INTEGRITY OF THE EVALUATION PROCESS.
    
    Parameters:
    - model (torch.nn.Module): The trained model to be used for predictions. It is imperative that this model is 
      compatible with the data provided by val_dataloader and can be moved to the specified device without modification.
    - val_dataloader (torch.utils.data.DataLoader): A DataLoader containing the validation dataset. It is crucial that
      the DataLoader returns batches of images along with their labels and image IDs as expected by this function.
    - device (torch.device): The device on which the computations will be performed. Typically, this would be a CUDA
      device for GPU acceleration. It is essential that the device specified here is compatible with the model and data.
    - results_dir (str): the directory to save the results.

    Outputs:
    - A CSV file named 'val_results.csv' is saved in the results directory. This file contains the image IDs,
      predicted classes, and probabilities for being in each class. DO NOT ALTER THE OUTPUT FORMAT OR FILE NAME.

    DO NOT MODIFY THIS FUNCTION. ANY CHANGES TO THIS FUNCTION MAY INVALIDATE THE EVALUATION PROCESS.
    """
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_img_id = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_dataloader)):
            img, labels = prepare_data(data, device)
            image_id = data['image_id']
            logits = model(img)
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_img_id.extend(image_id.numpy())
            all_probs.extend(probs)
            
        # Create a DataFrame
        df_results = pd.DataFrame({
            'Image_ID': all_img_id,
            'Predictions': all_preds,
            'Probability_Class_0': np.array(all_probs)[:, 0],
            'Probability_Class_1': np.array(all_probs)[:, 1]
        })
        
        results_csv_path = os.path.join(results_dir, 'val_results.csv')
        df_results.to_csv(results_csv_path, index=False)
    
        print(f'Results saved to {results_csv_path}')


def load_float_from_file(filename, default_value=0.0):
    """
    Loads a float value from a text file. Returns a default value if the file does not exist.

    Parameters:
    - filename (str): text file name.
    - default_value (float): value to return if the file does not exist.

    Outputs:
    - the file content as a float, or the default value.
    """
    try:
        with open(filename, 'r') as file:
            return float(file.read())
    except FileNotFoundError:
        return default_value
