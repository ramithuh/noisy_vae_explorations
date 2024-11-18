import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.example_libraries import optimizers


import os 

import flax.linen as nn
from flax.training import train_state

import torch
import optax
from datasets import load_dataset
import numpy as np
from PIL import Image
import orbax.checkpoint

# Enable TPU distribution
jax.distributed.initialize()  # Initialize JAX's distributed system

print("Number of devices:", jax.device_count())
print("Device type:", jax.devices()[0].platform)


from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

def custom_collate(batch):
    # Convert batch to JAX array directly
    return jnp.array(batch)
    
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

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda img: img.convert('RGB')),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

print("Loading data ...")
imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
imagenet_val = load_dataset('Maysee/tiny-imagenet', split='valid')


BATCH_SIZE = 4096

# Create datasets and dataloaders
train_dataset = ImageDataset(imagenet_train, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    collate_fn=custom_collate
)

BATCH_SIZE_PER_DEVICE = BATCH_SIZE // jax.device_count()

# VAE Model definition
class VAE(nn.Module):
    latent_dim: int = 128
    
    @nn.compact
    def __call__(self, x, deterministic=False):
        # Encoder
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        
        # Latent space
        mu = nn.Dense(features=self.latent_dim)(x)
        logvar = nn.Dense(features=self.latent_dim)(x)
        
        # Reparameterization
        if deterministic:
            z = mu
        else:
            std = jnp.exp(0.5 * logvar)
            key = self.make_rng('sampling')
            eps = random.normal(key, logvar.shape)
            z = mu + eps * std
        
        # Decoder
        x = nn.Dense(features=128 * 8 * 8)(z)
        x = x.reshape((-1, 8, 8, 128))
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.tanh(x)
        
        return x, mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = jnp.sum((recon_x - x) ** 2)
    kld = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar))
    return recon_loss + kld

# Initialize model and optimizer
def create_train_state(rng, learning_rate=1e-3):
    model = VAE()
    params = model.init(rng, jnp.ones((1, 64, 64, 3)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@jax.jit
def train_step(state, batch, rng):
    def loss_fn(params):
        # rng is now properly typed as a PRNGKey
        recon_x, mu, logvar = state.apply_fn(
            {'params': params},
            batch,
            deterministic=False,
            rngs={'sampling': rng}  # rng is now a proper PRNGKey
        )
        loss = vae_loss(recon_x, batch, mu, logvar)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train():
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    
    state = create_train_state(init_rng)
    state = jax.device_put_replicated(state, jax.devices())
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            # Convert PyTorch tensor to JAX array
            batch = jnp.array(batch)
            # Reshape for multiple devices
            batch = batch.reshape(jax.device_count(), -1, 64, 64, 3)
            
            # Create separate RNG keys for each device
            rng, step_rng = random.split(rng)
            step_rng = jax.random.split(step_rng, num=jax.device_count())
            
            state, loss = jax.pmap(train_step, axis_name='batch')(state, batch, step_rng)
            epoch_loss += jnp.mean(loss)
        
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")
        
if __name__ == '__main__':
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Start training
    train()