import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import os
import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda img: img.convert('RGB')),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

print("Loading data ...")
imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
imagenet_val = load_dataset('Maysee/tiny-imagenet', split='valid')


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self.transform(image)
        return image

BATCH_SIZE = 4096

train_dataset = ImageDataset(imagenet_train, transform=transform)
val_dataset = ImageDataset(imagenet_val, transform=transform)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False
)

class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(self.decoder_input(z).view(-1, 128, 8, 8))
        return decoded, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def _mp_fn(index):
    # Device setup
    device = xm.xla_device()
    
    # Model setup
    model = VAE().train().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Wrap data loader
    mp_device_loader = pl.MpDeviceLoader(train_loader, device)
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        
        for batch in mp_device_loader:
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(batch)  # Use model instead of vae
            loss = vae_loss(recon_batch, batch, mu, logvar)
            
            loss.backward()
            xm.optimizer_step(optimizer)
            train_loss += loss.item()
        
        # Print metrics only on master process
        if xm.is_master_ordinal():
            avg_loss = train_loss / len(train_loader.dataset)
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            xm.save(checkpoint, f"checkpoints/vae_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Launch training
    torch_xla.launch(_mp_fn, args=())