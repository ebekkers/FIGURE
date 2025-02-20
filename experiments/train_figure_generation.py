import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import imageio
from tqdm import tqdm

# Import data loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader_torch import FigureDataset
from models.unet import UNetModelWrapper

# ---- FLOW MATCHING MODEL ----
class FlowMatchingModel(nn.Module):
    def __init__(self, input_shape, num_channels):
        super().__init__()
        self.model = UNetModelWrapper(input_shape, num_channels, num_res_blocks=2)
    
    def forward(self, t, x):
        return self.model(t, x)

class FlowMatchingWrapper(nn.Module):
    def __init__(self, input_shape, num_channels):
        super().__init__()
        self.model = FlowMatchingModel(input_shape, num_channels)
    
    def forward(self, t, x):
        return self.model(t, x)

# ---- SAMPLER FUNCTION ----
def sample(model, device, input_shape, batch_size=64, steps=100, animation_gif="generated_animation.gif", final_samples_gif="generated_samples.gif"):
    """Euler integration from t=0 to t=1 and saves results as GIFs."""
    model.eval()
    frames = []
    with torch.no_grad():
        x_t = torch.randn(batch_size, *input_shape, device=device)
        dt = 1.0 / steps
        
        for i in range(steps):
            t = torch.full((batch_size,), i * dt, device=device)
            velocity = model(t, x_t)
            x_t = x_t + velocity * dt
            
            frame = ((x_t.cpu().numpy().transpose(0, 2, 3, 1) + 0.5) * 255).clip(0,255).astype('uint8')
            frames.append(frame[0])
    
    os.makedirs("results", exist_ok=True)
    imageio.mimsave(os.path.join("results", animation_gif), frames, fps=20, loop=0)
    print(f"Animation GIF saved to {animation_gif}")
    
    final_samples = ((x_t.cpu().numpy().transpose(0, 2, 3, 1) + 0.5) * 255).clip(0,255).astype('uint8')
    imageio.mimsave(os.path.join("results", final_samples_gif), final_samples, fps=1, loop=0)
    print(f"Final Samples GIF saved to {final_samples_gif}")

# ---- TRAINING FUNCTION ----
def train(model, train_loader, criterion, optimizer, scheduler, device, dataset_name, num_epochs, checkpoint_dir):
    print(f"Training Flow Matching Model on {dataset_name} for {num_epochs} epochs...")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for x_1, _, _, _ in loop:
            x_1 = x_1.to(device) - 0.5
            t = torch.rand(x_1.size(0), device=device)
            x_0 = torch.randn_like(x_1)
            x_t = x_0 * (1 - t.view(-1, 1, 1, 1)) + x_1 * t.view(-1, 1, 1, 1)
            
            optimizer.zero_grad()
            v_pred = model(t, x_t)
            v = (x_1 - x_0)
            loss = criterion(v_pred, v)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1}: Train Loss: {total_loss:.4f}")
        scheduler.step()
        print(f"Updated Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/flow_matching_epoch{epoch+1}.pth")
            print(f"✅ Saved checkpoint at epoch {epoch+1}!")
    
    print("🎉 Training complete!")

# ---- ARGUMENT PARSER ----
def main():
    parser = argparse.ArgumentParser(description="Train or sample from the Flow Matching model.")
    parser.add_argument("--dataset-name", type=str, default="FIGURE-Shape-F", help="Dataset name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num-channels", type=int, default=64, help="Number of channels in the model.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_generative", help="Directory to save checkpoints.")
    parser.add_argument("--sample-only", action="store_true", help="Skip training and only generate samples.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint for sampling.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (3, 64, 64)
    
    train_dataset = FigureDataset(args.dataset_name, split="train", download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = FlowMatchingWrapper(input_shape, args.num_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    if args.sample_only:
        if args.checkpoint is None:
            raise ValueError("You must provide a checkpoint path using --checkpoint when sampling only.")
        
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print(f"✅ Loaded model checkpoint from {args.checkpoint}")
        print("Generating samples...")
        sample(model, device, input_shape)
        print("Sampling complete!")
    else:
        train(model, train_loader, criterion, optimizer, scheduler, device, args.dataset_name, args.epochs, args.checkpoint_dir)
        print("Generating samples...")
        sample(model, device, input_shape)
        print("Sampling complete!")

if __name__ == "__main__":
    main()
