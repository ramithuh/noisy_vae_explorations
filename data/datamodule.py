import lightning as L
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from dataset import CustomImageNetDataset  # Importing the custom dataset class

class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, batch_size, cache_dir='./dataset_cache'):
        super().__init__()
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        # Transformation pipeline for Vision Transformers or CNN models
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 for ViT
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Repeat channels if grayscale
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])

    def setup(self, stage: str = None):
        # Load the dataset from the Hugging Face hub for each split
        if stage == 'fit' or stage is None:
            # Load train and validation datasets from Hugging Face
            dataset = load_dataset('Maysee/tiny-imagenet', cache_dir=self.cache_dir)
            self.imagenet_train = CustomImageNetDataset(dataset['train'], transform=self.transform)
            
            # Split the validation dataset into validation and test sets
            val_test_split = dataset['valid'].train_test_split(test_size=0.5, stratify_by_column='label')
            self.imagenet_val = CustomImageNetDataset(val_test_split['train'], transform=self.transform)
            self.imagenet_test = CustomImageNetDataset(val_test_split['test'], transform=self.transform)

        elif stage == 'test' or stage == 'predict':
            # Load only the valid split if testing or predicting
            dataset = load_dataset('Maysee/tiny-imagenet', split='valid', cache_dir=self.cache_dir)
            val_test_split = dataset.train_test_split(test_size=0.5, stratify_by_column='label')
            self.imagenet_test = CustomImageNetDataset(val_test_split['test'], transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=self.batch_size)
