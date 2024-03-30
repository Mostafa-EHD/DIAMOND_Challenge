from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from utils import read_image, augment_image

class DiamondDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and transforming Ultra-Wide Field (UWF) 2D retinal images for ci-DME prediction. 
    It supports reading images, resizing, optionally augmenting for training, and transforming to PyTorch tensors.

    Attributes:
        df (pandas.DataFrame): DataFrame containing the image paths, labels, and additional metadata.
        mode (str): Mode of operation. Can be 'train', 'val', or 'test' to apply appropriate transformations.
        args (argparse.Namespace): Command-line arguments object containing runtime parameters like root directory path.
        image_size (int): The size to which images are resized. Default is 448x448.
        transform (bool): Flag to determine whether to apply augmentation on the training dataset.

    Args:
        df (pandas.DataFrame): A DataFrame with at least 'image_path', 'next_DME_label', and 'image_id' columns.
        mode (str): The mode of operation - influences whether data augmentation is applied. Expected values are 'train', 'val', or 'test'.
        args (argparse.Namespace, optional): Arguments provided at runtime. This must include 'root_dir' if specified.
    """
    def __init__(self, df, mode='train', args=None):
        self.args = args
        self.mode = mode
        self.df = df.copy()
        self.image_size = 448  # Default image size to resize to
        self.transform = True  # By default, apply transformations

    def __len__(self):
        """
        Determine the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve an item by its index.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary containing the processed image tensor, its corresponding label, and the image ID.
                  The keys are 'image', 'label', and 'image_id'.
        """
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = row['next_DME_label']
        image_id = row['image_id']

        img_path = os.path.join(self.args.root_dir, image_path)

        image = read_image(img_path)
        image = image.resize((self.image_size, self.image_size))

        # Apply augmentation if in 'train' mode and transforms are enabled
        if self.mode == "train" and self.transform:
            image = augment_image(image)

        # Convert image to PyTorch tensor
        image = transforms.ToTensor()(image)

        return {'image': image, 'label': label, 'image_id': image_id}
