# test_imagenet_datamodule.py
from datamodule import ImageNetDataModule  # Make sure this matches the actual module file name
import torch

# Define parameters
BATCH_SIZE = 32

def test_setup():
    datamodule = ImageNetDataModule(batch_size=BATCH_SIZE)
    datamodule.setup(stage='fit')
    
    assert datamodule.imagenet_train is not None, "Train dataset not initialized"
    assert datamodule.imagenet_val is not None, "Validation dataset not initialized"
    assert datamodule.imagenet_test is not None, "Test dataset not initialized"
    print("Setup test passed.")

def test_train_dataloader():
    datamodule = ImageNetDataModule(batch_size=BATCH_SIZE)
    datamodule.setup(stage='fit')
    
    train_loader = datamodule.train_dataloader()
    assert isinstance(train_loader, torch.utils.data.DataLoader), "train_dataloader did not return a DataLoader"
    
    # Fetch a batch
    images, labels = next(iter(train_loader))
    print(labels)

    # Verify batch size and contents
    assert len(images) == BATCH_SIZE, f"Expected batch size {BATCH_SIZE}, but got {len(images)}"
    assert len(labels) == BATCH_SIZE, f"Expected batch size {BATCH_SIZE}, but got {len(labels)}"
    print("Train dataloader test passed.")

def test_val_dataloader():
    datamodule = ImageNetDataModule(batch_size=BATCH_SIZE)
    datamodule.setup(stage='fit')
    
    val_loader = datamodule.val_dataloader()
    assert isinstance(val_loader, torch.utils.data.DataLoader), "val_dataloader did not return a DataLoader"
    
    # Check the batch size and data format
    batch = next(iter(val_loader))
    assert len(batch[0]) == BATCH_SIZE, f"Expected batch size {BATCH_SIZE}, but got {len(batch[0])}"
    print("Validation dataloader test passed.")

def test_test_dataloader():
    datamodule = ImageNetDataModule(batch_size=BATCH_SIZE)
    datamodule.setup(stage='test')
    
    test_loader = datamodule.test_dataloader()
    assert isinstance(test_loader, torch.utils.data.DataLoader), "test_dataloader did not return a DataLoader"
    
    # Check the batch size and data format
    batch = next(iter(test_loader))
    assert len(batch[0]) == BATCH_SIZE, f"Expected batch size {BATCH_SIZE}, but got {len(batch[0])}"
    print("Test dataloader test passed.")

def test_predict_dataloader():
    datamodule = ImageNetDataModule(batch_size=BATCH_SIZE)
    datamodule.setup(stage='predict')
    
    predict_loader = datamodule.predict_dataloader()
    assert isinstance(predict_loader, torch.utils.data.DataLoader), "predict_dataloader did not return a DataLoader"
    
    # Check the batch size and data format
    batch = next(iter(predict_loader))
    assert len(batch[0]) == BATCH_SIZE, f"Expected batch size {BATCH_SIZE}, but got {len(batch[0])}"
    print("Predict dataloader test passed.")

if __name__ == "__main__":
    print("Running tests...")
    test_setup()
    test_train_dataloader()
    test_val_dataloader()
    test_test_dataloader()
    test_predict_dataloader()
    print("All tests passed.")
