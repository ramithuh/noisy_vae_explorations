import lightning as L
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms


class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # def prepare_data(self):
    #     # download
    #     load_dataset('Maysee/tiny-imagenet', split='train')
    #     load_dataset('Maysee/tiny-imagenet', split='valid')

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        
        self.imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
        imagenet_val_combined = load_dataset('Maysee/tiny-imagenet', split='valid')
        
        imagenet_val_test = imagenet_val_combined.train_test_split(test_size=0.5, stratify_by_column='label')
        self.imagenet_val = imagenet_val_test['train']
        self.imagenet_test = imagenet_val_test['test']
        

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=32)