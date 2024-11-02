import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomImageNetDataset(Dataset):
    def __init__(self, huggingface_dataset, transform=None):
        """
        Args:
            huggingface_dataset: Our ImageNet dataset from Hugging Face
            transform: Transformations to apply to the images
        """
        self.dataset = huggingface_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve the image and label
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']

        # Convert image to PIL format if it's not already

        # Apply the transform if specified
        if self.transform:
            image = self.transform(image)

        return image, label
