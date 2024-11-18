import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

import os
import wandb
from tqdm import tqdm

# Initialize wandb
wandb.init(
    project="ML_10701_Project",
    entity="ramith-mit",
    config={
        "architecture": "VAE",
        "dataset": "tiny-imagenet",
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "latent_dim": 128
    }
)

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

train_dataset = ImageDataset(imagenet_train, transform=transform)
val_dataset = ImageDataset(imagenet_val, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

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

# Training loop
device = 'cuda:0' #torch.device("mps" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Watch the model with wandb
wandb.watch(vae, log="all", log_freq=10)

# Create directory for saving checkpoints
os.makedirs("checkpoints", exist_ok=True)



epochs = 100
for epoch in tqdm(range(epochs), desc='Epochs'):
    vae.train()
    train_loss = 0
    batch_count = 0
    
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(batch)
        loss = vae_loss(recon_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        batch_count += 1

        # Log batch loss
        wandb.log({
            "batch_loss": loss.item(),
            "epoch": epoch,
            "batch": batch_count
        })

        optimizer.step()
    
    # Calculate average epoch loss
    avg_epoch_loss = train_loss / len(train_loader.dataset)
    
    # Log epoch metrics
    wandb.log({
        "epoch": epoch,
        "avg_train_loss": avg_epoch_loss,
    })
    
    # Save checkpoint locally
    checkpoint_path = f"checkpoints/vae_epoch_{epoch+1}.pth"
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss
    }
    torch.save(checkpoint, checkpoint_path)
    
    # Log model checkpoint to wandb
    wandb.save(checkpoint_path)
    
    print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}')

# Close wandb run
wandb.finish()