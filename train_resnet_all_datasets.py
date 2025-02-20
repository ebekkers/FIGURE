import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader_torch import FigureDataset
import json


# ---- TRAINING LOOP ----
def train(model, train_loader, test_loader, test_bias_loader, has_test_bias, optimizer, scheduler, criterion, num_epochs, validate_every_epoch):
    print(f"Training on {DATASET_NAME} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)

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

            loop.set_postfix(loss=loss.item(), acc=correct / total)

        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Step the scheduler
        scheduler.step()
        print(f"Updated Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        if validate_every_epoch:
            # Evaluate after each epoch
            evaluate(model, test_loader, "Test Set")
            if has_test_bias:
                evaluate(model, test_bias_loader, "Bias-Swapped Test Set")

    print("🎉 Training complete!")


# ---- EVALUATION FUNCTION ----
def evaluate(model, loader, name):
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

    # ---- CONFIGURATION ----
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VALIDATE_EVERY_EPOCH = False

    results = {}

    for consistency_str in ["1.0", "0.99", "0.9", "0.75", "0.5", "0.25"]:
        results[consistency_str] = {}
        for DATASET_STR in ["B", "CB", "PI", "F"]:
            DATASET_NAME = "FIGURE-Shape-" + DATASET_STR
            print(f"\n\nTraining with color consistency: {consistency_str}")
            # Load datasets
            color_consistency = float(consistency_str)
            train_dataset = FigureDataset(DATASET_NAME, split="train", download=True, color_consistency=color_consistency)
            test_dataset = FigureDataset(DATASET_NAME, split="test", download=True, color_consistency=color_consistency)

            # Load bias-swapped test set if available
            if DATASET_STR in ["CB", "F"]:
                test_bias_dataset = FigureDataset(DATASET_NAME, split="test-bias", download=True, color_consistency=color_consistency)
                has_test_bias = True
            else:
                test_bias_dataset = None
                has_test_bias = False

            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            test_bias_loader = DataLoader(test_bias_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) if has_test_bias else None

            # ---- MODEL ----
            class SimpleCNN(nn.Module):
                """A simple CNN model with a ResNet-like structure."""
                def __init__(self, num_classes=4):
                    super(SimpleCNN, self).__init__()
                    self.backbone = models.resnet18(pretrained=True)
                    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

                def forward(self, x):
                    return self.backbone(x)


            # Initialize model
            model = SimpleCNN(num_classes=4).to(DEVICE)

            # ---- TRAINING SETUP ----
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Cosine Annealing LR Scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

            # Train the model
            train(model, train_loader, test_loader, test_bias_loader, has_test_bias, optimizer, scheduler, criterion, NUM_EPOCHS, VALIDATE_EVERY_EPOCH)

            # Evaluate
            print("\nFinal Test Set Accuracy:")
            test_acc = evaluate(model, test_loader, "Test Set")
            results[consistency_str][DATASET_STR] = test_acc
            if has_test_bias:
                print("Final Bias-Swapped Test Set Accuracy:")
                test_acc_ood = evaluate(model, test_bias_loader, "Bias-Swapped Test Set")
                results[consistency_str][DATASET_STR + "-OOD"] = test_acc_ood

    # Save results as JSON
    results_path = "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")

    

    # Load results
    results_path = "results.json"
    with open(results_path, "r") as f:
        results = json.load(f)

    # Generate Markdown
    md_content = "# Experiment Results: ResNet on FIGURE Dataset\n\n"
    md_content += "| Consistency | B | CB | CB (OOD) | PI | F | F (OOD) |\n"
    md_content += "|------------|----|----|----------|----|----|--------|\n"

    for consistency, res in results.items():
        B = res.get("B", "N/A")
        CB = res.get("CB", "N/A")
        CB_OOD = res.get("CB-OOD", "N/A")
        PI = res.get("PI", "N/A")
        F = res.get("F", "N/A")
        F_OOD = res.get("F-OOD", "N/A")

        md_content += f"| {consistency} | {B:.4f} | {CB:.4f} | {CB_OOD:.4f} | {PI:.4f} | {F:.4f} | {F_OOD:.4f} |\n"

    # Save Markdown file
    md_file = "experiment_results.md"
    with open(md_file, "w") as f:
        f.write(md_content)

    print(f"Markdown report saved to {md_file}")
