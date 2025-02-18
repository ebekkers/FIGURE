import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader_torch import FigureDataset

# ---- CONFIGURATION ----
DATASET_NAME = "FIGURE-Shape-PI"  # Change to any dataset
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load datasets
train_dataset = FigureDataset(DATASET_NAME, split="train", download=True)
test_dataset = FigureDataset(DATASET_NAME, split="test", download=True)

# Make things more challenging by reducing train set size to 10%
# from torch.utils.data import Subset
# subset_size = int(0.1 * len(train_dataset))  # 10% of training set
# subset_indices = torch.randperm(len(train_dataset))[:subset_size]  # Random subset
# train_dataset = Subset(train_dataset, subset_indices)

# Load bias-swapped test set if available
try:
    test_bias_dataset = FigureDataset(DATASET_NAME, split="test-bias", download=True)
    has_test_bias = True
except RuntimeError:
    print(f"No bias-swapped test set found for {DATASET_NAME}.")
    has_test_bias = False

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_bias_loader = DataLoader(test_bias_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) if has_test_bias else None

# ---- MODEL: ----
class SimpleCNN(nn.Module):
    """A simple CNN model with a ResNet-like structure."""
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Select model type
MODEL_TYPE = "cnn" 
model = SimpleCNN(num_classes=4)
model = model.to(DEVICE)


# ---- TRAINING SETUP ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---- TRAINING LOOP ----
def train():
    print(f"Training {MODEL_TYPE.upper()} on {DATASET_NAME} for {NUM_EPOCHS} epochs...")
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=True)

        for images, _, _, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update tqdm bar with real-time loss
            loop.set_postfix(loss=loss.item(), acc=correct / total)

        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Evaluate after each epoch with progress bars
        test_acc = evaluate(test_loader, "Test Set")
        test_bias_acc = evaluate(test_bias_loader, "Bias-Swapped Test Set") if has_test_bias else None

        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{DATASET_NAME}_{MODEL_TYPE}_best.pth")
            print("✅ Saved best model checkpoint!")

    print("🎉 Training complete!")


# ---- EVALUATION FUNCTION ----
def evaluate(loader, name):
    if loader is None:
        return None

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Evaluating {name}", leave=False)
        
        for images, _, _, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"📊 {name} Accuracy: {acc:.4f}")
    return acc


# ---- RUN TRAINING ----
if __name__ == "__main__":
    train()
    print("Final Test Set Accuracy:")
    evaluate(test_loader, "Test Set")
    
    if has_test_bias:
        print("Final Bias-Swapped Test Set Accuracy:")
        evaluate(test_bias_loader, "Bias-Swapped Test Set")
