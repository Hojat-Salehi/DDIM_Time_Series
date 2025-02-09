import os
import math
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import GPUtil
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def setup_device(device_type: str):
    if device_type == "GPU":
        try:
            return select_free_gpu(max_memory_usage=0.5)
        except Exception as e:
            print(f"Error selecting GPU: {e}")
            print("No free GPU available. Selecting CPU for processing.")
            return torch.device("cpu")
    return torch.device("cpu")

def select_free_gpu(max_memory_usage=0.5, priority="memory"):
    """
    Select a GPU whose memory usage is less than `max_memory_usage`.
    Among those, either pick the GPU with the lowest load (priority="load")
    or the one with the lowest memory usage (priority="memory").
    If no GPU satisfies the memory criterion, pick the GPU with the smallest
    (memoryUtil, load) among all GPUs.
    """
    devices = GPUtil.getGPUs()
    # Filter out GPUs that exceed the memory threshold
    available_gpus = [i for i in range(len(devices)) if devices[i].memoryUtil < max_memory_usage]

    if not available_gpus:
        print(f"No available GPU with memory usage < {max_memory_usage*100:.0f}%.")
        # Fallback: pick GPU with smallest memory usage, then load
        selected_gpu = sorted(devices, key=lambda x: (x.memoryUtil, x.load))[0].id
    else:
        print(f"GPUs below {max_memory_usage*100:.0f}% memory usage: {available_gpus}")
        if priority == "memory":
            # Pick the GPU with the *lowest memory usage* among the filtered set
            mem_util_values = [devices[i].memoryUtil for i in available_gpus]
            min_mem = min(mem_util_values)
            min_mem_index = mem_util_values.index(min_mem)
            selected_gpu = available_gpus[min_mem_index]
        else:
            # Priority="load": pick the GPU with the *lowest load*
            gpu_loads = [devices[i].load for i in available_gpus]
            min_load = min(gpu_loads)
            min_load_index = gpu_loads.index(min_load)
            selected_gpu = available_gpus[min_load_index]
    
    chosen = devices[selected_gpu]
    print(f"Selected GPU: {selected_gpu} "
          f"(memory={chosen.memoryUtil*100:.1f}%, "
          f"load={chosen.load*100:.1f}%)")
    
    return torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")

device = setup_device("GPU")

############################################
# Dataset Class
############################################
class StockImageDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, train_ratio=0.9):
        self.root_dir = root_dir
        self.transform = transform
        # Get sorted list of images
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'stock_*.png')))
        total_images = len(self.image_paths)
        train_count = int(total_images * train_ratio)
        if train:
            self.image_paths = self.image_paths[:train_count]
        else:
            self.image_paths = self.image_paths[train_count:]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')  # RGB image
        if self.transform is not None:
            img = self.transform(img)
        # img: (C=3, H, W)
        return img



class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        # print(f"\n=== Block Initialization ===")
        # print(f"Input channels: {in_ch}")
        # print(f"Output channels: {out_ch}")
        # print(f"Time embedding dim: {time_emb_dim}")
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        
        # print(f"Creating conv1 with in_ch={in_ch}, out_ch={out_ch}")
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        # print(f"Creating conv2 with in_ch={out_ch}, out_ch={out_ch}")
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        # if in_ch != out_ch:
        #     print(f"Creating skip connection conv with in_ch={in_ch}, out_ch={out_ch}")
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # print("=== Block Initialization Complete ===\n")

    def forward(self, x, time_emb=None):
        # print(f"\n=== Block Forward Pass ===")
        # print(f"Input tensor shape: {x.shape}")
        # print(f"Block's configured in_ch: {self.in_ch}")
        # print(f"Block's configured out_ch: {self.out_ch}")
        
        if x.shape[1] != self.in_ch:
            print(f"WARNING: Channel mismatch!")
            print(f"Expected {self.in_ch} channels, got {x.shape[1]} channels")
            
        h = self.conv1(x)
        # print(f"After conv1 shape: {h.shape}")
        h = self.norm1(h)
        h = F.silu(h)
        
        if self.time_mlp and time_emb is not None:
            time_emb = self.time_mlp(F.silu(time_emb))
            # print(f"Time embedding shape: {time_emb.shape}")
            h = h + time_emb[..., None, None]
            
        h = self.conv2(h)
        # print(f"After conv2 shape: {h.shape}")
        h = self.norm2(h)
        h = F.silu(h)
        
        skip_out = self.skip(x)
        # print(f"Skip connection output shape: {skip_out.shape}")
        
        out = h + skip_out
        # print(f"Final output shape: {out.shape}")
        # print("=== Block Forward Pass Complete ===\n")
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # print(f"\nInitializing AttentionBlock with channels={channels}")
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # print(f"\nAttentionBlock forward - Input shape: {x.shape}")
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        scale = 1 / math.sqrt(math.sqrt(C))
        attn = torch.einsum('bci,bcj->bij', q.view(B, C, -1) * scale, k.view(B, C, -1) * scale)
        attn = attn.softmax(dim=-1)
        
        h = torch.einsum('bij,bcj->bci', attn, v.view(B, C, -1))
        h = h.view(B, C, H, W)
        out = self.proj(h) + x
        # print(f"AttentionBlock output shape: {out.shape}")
        return out

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=256,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        attention_resolutions=(8, 16),
    ):
        super().__init__()
        # print(f"\nInitializing UNet with:")
        # print(f"in_channels={in_channels}, out_channels={out_channels}")
        # print(f"base_channels={base_channels}")
        # print(f"channel_mults={channel_mults}")
        
        # Time embedding and initial conv remain the same
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.downs = nn.ModuleList([])
        channels = []  # Track channels for skip connections
        in_ch = base_channels
        
        # print("\nDownsampling path:")
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            # print(f"\nLevel {level} - in_ch: {in_ch}, out_ch: {out_ch}")
            
            for _ in range(2):
                self.downs.append(Block(in_ch, out_ch, time_emb_dim))
                channels.append(out_ch)  # Keep track of channels for skip connections
                if out_ch in attention_resolutions:
                    self.downs.append(AttentionBlock(out_ch))
                in_ch = out_ch
            
            if level != len(channel_mults) - 1:
                self.downs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
                # Removed channels.append(out_ch) from here
        
        # Middle
        mid_channels = base_channels * channel_mults[-1]
        # print(f"\nMiddle - channels: {mid_channels}")
        self.middle = nn.ModuleList([
            Block(mid_channels, mid_channels, time_emb_dim),
            AttentionBlock(mid_channels),
            Block(mid_channels, mid_channels, time_emb_dim)
        ])
        
        # Upsampling
        self.ups = nn.ModuleList([])
        # print("\nUpsampling path:")
        channels = channels[::-1]  # Reverse channels for skip connections
        in_ch = mid_channels
        
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            # print(f"\nLevel {level} - out_ch: {out_ch}")
            
            # First block: input channels = current + skip
            skip_ch = channels.pop(0)
            up_in_ch = in_ch + skip_ch
            # print(f"First block - in_ch: {up_in_ch} ({in_ch} + {skip_ch}), out_ch: {out_ch}")
            self.ups.append(Block(up_in_ch, out_ch, time_emb_dim))
            
            if out_ch in attention_resolutions:
                self.ups.append(AttentionBlock(out_ch))
            
            # Second block: input channels = current + skip
            skip_ch = channels.pop(0)
            up_in_ch = out_ch + skip_ch
            # print(f"Second block - in_ch: {up_in_ch} ({out_ch} + {skip_ch}), out_ch: {out_ch}")
            self.ups.append(Block(up_in_ch, out_ch, time_emb_dim))
            
            if out_ch in attention_resolutions:
                self.ups.append(AttentionBlock(out_ch))
            
            if level != 0:
                self.ups.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))
            
            in_ch = out_ch
        
        # Final block
        final_skip_ch = channels[0] if channels else base_channels
        # print(f"\nFinal block - in_ch: {in_ch + final_skip_ch} ({in_ch} + {final_skip_ch}), out_ch: {base_channels}")
        self.final = nn.Sequential(
            Block(in_ch + final_skip_ch, base_channels, time_emb_dim),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
        # print("\nInitialization complete")

    def forward(self, x, time):
        # print("\n=== Starting UNet Forward Pass ===")
        
        # Time embedding
        t = self.time_mlp(time)
        # print(f"Time embedding shape: {t.shape}")
        
        # Initial conv
        h = self.init_conv(x)
        # print(f"After initial conv: {h.shape}")
        
        # Store skip connections
        skips = []
        first_skip = None
        
        # Downsampling
        # print("\n=== Starting Downsampling Path ===")
        skips = []
        for i, layer in enumerate(self.downs):
            # print(f"\n--- Downsampling Layer {i} ---")
            # print(f"Current feature map shape: {h.shape}")
            
            if isinstance(layer, Block):
                h = layer(h, t)
                if first_skip is None:
                    first_skip = h
     
                skips.append(h)
                # print(f"Stored skip connection with shape: {h.shape}")
                
            elif isinstance(layer, nn.Conv2d):
                # print("Processing downsampling conv")
                # print(f"Conv weight shape: {layer.weight.shape}")
                h = layer(h)
                # print(f"After downsampling shape: {h.shape}")
        
        # Middle
        # print("\n=== Middle Blocks ===")
        for i, layer in enumerate(self.middle):
            if isinstance(layer, Block):
                # print(f"\nMiddle Block {i}")
                h = layer(h, t)
            else:
                # print(f"\nMiddle Attention {i}")
                h = layer(h)
            # print(f"After middle block {i}, shape: {h.shape}")
        
        # Upsampling
        # Upsampling
        # print("\n=== Starting Upsampling Path ===")
        for i, layer in enumerate(self.ups):
            # print(f"\n--- Upsampling Layer {i} ---")
            # print(f"Current feature map shape: {h.shape}")
            
            if isinstance(layer, nn.ConvTranspose2d):
                # print("Processing ConvTranspose2d layer")
                # print(f"Layer weight shape: {layer.weight.shape}")
                h = layer(h)
                # print(f"After upsampling shape: {h.shape}")
            
            elif isinstance(layer, Block):
                # print("Processing Block layer")
                if len(skips) > 0:
                    skip = skips.pop()
                    # print(f"Skip connection shape: {skip.shape}")
                    # print(f"Current feature map shape before concat: {h.shape}")
                    h = torch.cat([h, skip], dim=1)
                    # print(f"After concatenation shape: {h.shape}")
                    # print(f"Block's expected input channels: {layer.in_ch}")
                h = layer(h, t)
                # print(f"After block processing shape: {h.shape}")
            
            elif isinstance(layer, AttentionBlock):
                # print("Processing AttentionBlock layer")
                # print(f"Input shape to attention: {h.shape}")
                h = layer(h)
                # print(f"After attention shape: {h.shape}")

        # Final
        # print("\n=== Final Processing ===")
        # print(f"Before final processing, shape: {h.shape}")
        # if len(skips) > 0:
        # skip = skips.pop()
        h = torch.cat([h, first_skip], dim=1)
        # print(f"After final concatenation: {h.shape}")
        h = self.final(h)
        # print(f"Final output shape: {h.shape}")
        
        return h

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def get_beta_schedule(schedule_type, n_timesteps):
    if schedule_type == "linear":
        return torch.linspace(0.0001, 0.02, n_timesteps)
    elif schedule_type == "cosine":
        steps = n_timesteps + 1
        x = torch.linspace(0, n_timesteps, steps)
        alphas_cumprod = torch.cos(((x / n_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_type}")

def validate_diffusion_parameters(betas, alphas, alphas_cumprod):
    assert torch.all(betas > 0) and torch.all(betas <= 1), "Betas must be in (0, 1]"
    assert torch.all(alphas >= 0) and torch.all(alphas <= 1), "Alphas must be in [0, 1]"
    assert torch.all(alphas_cumprod >= 0) and torch.all(alphas_cumprod <= 1), "Alpha_cumprod must be in [0, 1]"
    assert torch.all(torch.diff(alphas_cumprod) <= 0), "Alpha_cumprod must be monotonically decreasing"

def setup_diffusion_parameters(n_timesteps, schedule_type="linear", device='cpu'):
    # Get beta schedule
    betas = get_beta_schedule(schedule_type, n_timesteps).to(device)
    
    # Calculate alpha values (α_t = 1 - β_t)
    alphas = (1 - betas).to(device)
    
    # Calculate cumulative product of alphas (ᾱ_t = Π_{s=1}^t α_s)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Store previous timestep's alpha_cumprod for sampling
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    # Calculate other helper values used in sampling
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # Validate parameters
    validate_diffusion_parameters(betas, alphas, alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas
    }

model = UNet(
    in_channels=3,
    out_channels=3,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    attention_resolutions=(8, 16),
    time_emb_dim=256
)



# Define parameters
batch_size = 4
image_size = 224
n_timesteps = 1000  # Number of diffusion steps

# Load your dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Use your existing dataset class
train_dataset = StockImageDataset('stock_images', train=True, transform=transform)
test_dataset = StockImageDataset('stock_images', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



diffusion_params = setup_diffusion_parameters(
    n_timesteps=1000,
    schedule_type="linear",
    device=device
)


# Base q_sample function for the diffusion process
def q_sample(x_start, t, noise=None, diffusion_params=None, device='cpu'):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(diffusion_params['sqrt_alphas_cumprod'], t, x_start.shape, device)
    sqrt_one_minus_alphas_cumprod_t = extract(diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x_start.shape, device)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# Conditional q_sample that only applies diffusion to the third channel
def q_sample_conditional(x_start, t, noise=None, diffusion_params=None, device='cpu'):
    """
    Apply diffusion only to the third channel while keeping the first two channels unchanged.
    This is an extension to the original DDIM paper for conditional generation.
    """
    if noise is None:
        noise = torch.randn_like(x_start[:, 2:3]).to(device)
    
    # Keep first two channels unchanged
    unchanged_channels = x_start[:, :2]
    
    # Apply diffusion only to third channel
    noised_channel = q_sample(x_start[:, 2:3], t, noise, diffusion_params, device)
    
    # Concatenate back all three channels
    result = torch.cat([unchanged_channels, noised_channel], dim=1)
    
    assert result.shape[1] == 3, f"Expected 3 channels, got {result.shape[1]}"
    return result.to(device)
# Helper function to extract appropriate timestep values
# Improved extract function
def extract(a, t, x_shape, device=None):
    batch_size = t.shape[0]
    out = a.gather(-1, t).float()
    if device is not None:
        out = out.to(device)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# Move model to device
model = model.to(device)

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)



# Improved loss function
def get_loss(model_output, noise, loss_scale=1.0):
    return F.mse_loss(model_output, noise, reduction='sum') * loss_scale # Sum instead of mean


# Training loop parameters
n_epochs = 2000
save_every = 10  # Save model every 10 epochs

# Lists to store losses
train_losses = []
num_checkpoints_to_keep=5
checkpoint_history = deque(maxlen=num_checkpoints_to_keep)

for epoch in range(n_epochs):
    model.train()
    epoch_losses = []
    
    for batch in train_loader:
        # Move batch to device
        batch = batch.to(device)
        # print("Original batch shape:", batch.shape)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Sample random timesteps
        t = torch.randint(0, n_timesteps, (batch.size(0),), device=device).long()
        
        # Generate random noise only for the third channel
        noise = torch.randn_like(batch[:, 2:3]).to(device)
        
        # Get noisy image (only third channel is noised)
        # noised_batch = q_sample_conditional(batch, t, noise, device)
        # Use conditional sampling with new parameters
        noised_batch = q_sample_conditional(
            batch, t, noise, 
            diffusion_params=diffusion_params,
            device=device
        )
        # print(noised_batch.shape)
        # Predict noise
        # predicted_noise = model(noised_batch, t).to(device)
        # print(predicted_noise.shape)
        predicted_noise = model(noised_batch, t)[:, 2:3]  # Only take third channel output
        
        # Calculate loss only on third channel
        loss = get_loss(predicted_noise, noise)
        # print(loss)
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        # print(loss)
        
        epoch_losses.append(loss.item())
    
    # Calculate average loss for the epoch
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(avg_loss)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model checkpoint
    if (epoch + 1) % save_every == 0:
        # Create checkpoint filename with timestamp for better tracking
        checkpoint_filename = os.path.join(
            checkpoint_dir, 
            f'model_checkpoint_epoch_{epoch+1}.pt'
        )
        
        # Save the checkpoint
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_losses': train_losses,  # Also saving training history
            }, checkpoint_filename)
            print(f"Checkpoint saved successfully at epoch {epoch+1}")
        except Exception as e:
            print(f"Error saving checkpoint at epoch {epoch+1}: {e}")
        # Keep track of checkpoints using a deque with max length
        

        # Append to history
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        checkpoint_history.append(checkpoint_path)

        # If we exceed the limit, remove oldest checkpoint
        if len(checkpoint_history) == num_checkpoints_to_keep and os.path.exists(checkpoint_history[0]):
            try:
                os.remove(checkpoint_history[0])
                print(f"Removed old checkpoint: {checkpoint_history[0]}")
            except Exception as e:
                print(f"Error removing checkpoint: {e}")

# Improved sampling timestep function
def get_sampling_timesteps(total_timesteps=1000, sampling_steps=50, device='cuda', spacing='linear'):
    if spacing == 'linear':
        return torch.linspace(0, total_timesteps-1, sampling_steps, dtype=torch.long, device=device)
    elif spacing == 'quadratic':
        steps = torch.linspace(0, (total_timesteps-1)**2, sampling_steps, device=device)
        return torch.floor(steps.sqrt()).long()
    else:
        raise ValueError(f"Unknown spacing type: {spacing}")



@torch.no_grad()
def sample_ddim(model, x_cond, diffusion_params, n_inference_steps=50, total_timesteps=1000, device='cuda', eta=0.0):
    """
    Conditional DDIM sampling where only the third channel is generated and the first two channels
    are kept as conditioning information.
    Args:
        x_cond: Conditioning information in first two channels
        diffusion_params: Dictionary of diffusion parameters
        n_inference_steps: Number of sampling steps (can be much less than training steps)
        total_timesteps: Total number of timesteps used in training
        eta: Controls the stochasticity (0 = DDIM, 1 = DDPM)
    """
    # Get sampling timesteps
    timesteps = get_sampling_timesteps(total_timesteps, n_inference_steps, device)
    
    # Initialize x_t with noise only for the third channel
    x_t = torch.randn_like(x_cond[:, 2:3])
    # Concatenate with conditioning channels
    x_t = torch.cat([x_cond[:, :2], x_t], dim=1)
    
    # Store intermediate steps
    intermediate_images = []
    intermediate_images.append(x_t.cpu().clone())
    timesteps = timesteps.flip(0)
    for i in range(len(timesteps)-1):
        t = timesteps[i]
        t_prev = timesteps[i+1]
        
        # Expand t for batch dimension
        t_batch = t.expand(x_t.shape[0])
        
        # Predict noise
        predicted_noise = model(x_t, t_batch)
        
        # Get alpha values for current and previous timestep
        alpha_t = diffusion_params['alphas_cumprod'][t]
        alpha_t_prev = diffusion_params['alphas_cumprod'][t_prev]
        
        # Predict x0 (only for third channel)
        pred_x0 = torch.cat([
            x_cond[:, :2],
            (x_t[:, 2:3] - diffusion_params['sqrt_one_minus_alphas_cumprod'][t] * predicted_noise[:, 2:3]) / 
            diffusion_params['sqrt_alphas_cumprod'][t]
        ], dim=1)

        # DDIM sigma computation
        sigma_t = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        
        # Compute predicted direction
        pred_dir = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * predicted_noise[:, 2:3]
        
        # Get x_{t-1}
        x_t_new = torch.sqrt(alpha_t_prev) * pred_x0[:, 2:3] + pred_dir
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x_t[:, 2:3])
            x_t_new = x_t_new + sigma_t * noise
        
        # Only update the third channel
        x_t = x_t.clone()
        x_t[:, 2:3] = x_t_new
        
        # Store intermediate step
        if i % (n_inference_steps // 10) == 0:
            intermediate_images.append(x_t.cpu().clone())
    
    intermediate_images.append(x_t.cpu().clone())
    
    return x_t, intermediate_images


@torch.no_grad()
def test_and_visualize(model, test_loader, diffusion_params, save_dir, num_samples=10, n_inference_steps=50, device='cuda'):
    """
    Test the model and save visualizations of original vs predicted images.
    Args:
        model: The trained model
        test_loader: DataLoader for test set
        save_dir: Directory to save visualizations
        num_samples: Number of test samples to process
        n_inference_steps: Number of DDIM sampling steps
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'comparisons'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'intermediate_steps'), exist_ok=True)
    
    model.eval()
    test_iter = iter(test_loader)
    
    for idx in range(num_samples):
        try:
            # Get a test sample
            test_batch = next(test_iter).to(device)
            
            # # Sample using DDIM
            # final_pred, intermediate_steps = sample_ddim(
            #     model, 
            #     test_batch,
            #     n_inference_steps=n_inference_steps,
            #     device=device
            # )
            # Sampling
            final_pred, intermediate_steps = sample_ddim(
                model=model,
                x_cond=test_batch,  # Your first two channels for conditioning
                diffusion_params=diffusion_params,
                n_inference_steps=50,
                total_timesteps=1000,
                device=device,
                eta=0.0  # 0 for deterministic DDIM, 1 for stochastic DDPM
            )
            
            # Create figure for this sample
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            show_sample(test_batch[0].cpu(), ax=axes[0])
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Final prediction
            show_sample(final_pred[0].cpu(), ax=axes[1])
            axes[1].set_title('Predicted')
            axes[1].axis('off')
            
            # Difference (optional)
            diff = torch.abs(test_batch[0, 2:3] - final_pred[0, 2:3])  # Only third channel
            show_sample(diff.cpu(), ax=axes[2])
            axes[2].set_title('Difference (Channel 3)')
            axes[2].axis('off')
            
            # Save comparison plot
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'comparisons', f'sample_{idx}.png'))
            plt.close()
            
            # Save intermediate steps (as a grid)
            if len(intermediate_steps) > 0:
                fig_steps, axes_steps = plt.subplots(1, len(intermediate_steps), figsize=(20, 4))
                for step_idx, step_img in enumerate(intermediate_steps):
                    show_sample(step_img[0].cpu(), ax=axes_steps[step_idx])
                    axes_steps[step_idx].set_title(f'Step {step_idx}')
                    axes_steps[step_idx].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'intermediate_steps', f'sample_{idx}_steps.png'))
                plt.close()
            
            print(f"Processed sample {idx + 1}/{num_samples}")
            
        except StopIteration:
            print("Reached end of test dataset")
            break

def show_sample(img_tensor, ax=None):
    """Helper function to show an image"""
    if ax is None:
        ax = plt.gca()
    
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img + 1) / 2.0  # Denormalize
    ax.imshow(img)
    return ax

# Usage example:
save_directory = "test_results"
test_and_visualize(
    model=model,
    test_loader=test_loader,
    diffusion_params=diffusion_params,
    save_dir=save_directory,
    num_samples=10,  # Number of test samples to process
    n_inference_steps=50,  # Number of DDIM sampling steps
    device=device
)




