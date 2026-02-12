# train.py - Complete MNIST Neural Network Training Script
# Step-by-step implementation from scratch

# ============================================
#  STEP 1 - Imports
# ============================================
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ============================================
#  STEP 2 - Set Device (CPU for now, GPU ready)
# ============================================
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Uncomment for GPU
print(f"Using device: {device}")

# ============================================
#  STEP 3 - Define Hyperparameters
# ============================================
batch_size = 64
learning_rate = 0.001
epochs = 5

# ============================================
#  STEP 4 - Load MNIST Dataset
# ============================================
print("\n Loading MNIST dataset...")

# 4.1 Define the transform - convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL image to tensor [0,1]
])

# 4.2 Load training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# 4.3 Load test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

# 4.4 Create DataLoaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 4.5 Verify data loading
print(f" Training dataset: {len(train_dataset)} images")
print(f" Test dataset: {len(test_dataset)} images")
print(f" Training batches: {len(train_loader)}")
print(f" Test batches: {len(test_loader)}")

# Check one batch
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"\n Batch shape: {images.shape}")  # [64, 1, 28, 28]
print(f" Labels shape: {labels.shape}")   # [64]
print(f" Pixel range: [{images.min():.3f}, {images.max():.3f}]")

# ============================================
#  STEP 5 - Define Neural Network Class
# ============================================
print("\n Building neural network...")

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        # Input: 28x28x1 (784 features)
        # Hidden layers: 128 -> 64
        # Output: 10 classes (digits 0-9)
        
        self.network = nn.Sequential(
            # Flatten the 28x28 image to a 784 vector
            nn.Flatten(),
            
            # First hidden layer: 784 -> 128
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            
            # Second hidden layer: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # Output layer: 64 -> 10
            nn.Linear(64, 10)
            # No softmax here because CrossEntropyLoss includes it
        )
    
    def forward(self, x):
        return self.network(x)

# Create model instance and move to device
model = MNISTClassifier().to(device)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n Model parameters: {total_params:,} total, {trainable_params:,} trainable")

# ============================================
#  STEP 6 - Define Loss Function and Optimizer
# ============================================
criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax + NLLLoss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"\n Loss function: CrossEntropyLoss")
print(f" Optimizer: Adam (lr={learning_rate})")

# ============================================
#  STEP 7 - Training Loop
# ============================================
print("\n Starting training...")
print("-" * 60)

for epoch in range(epochs):
    # Set model to training mode
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Loop through batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print every 200 batches
        if (batch_idx + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Accuracy: {100 * correct/total:.2f}%')
    
    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'\n Epoch [{epoch+1}/{epochs}] - '
          f'Average Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%')
    print("-" * 60)

print(" Training completed!")

# ============================================
#  STEP 8 - Test the Model
# ============================================
print("\n Testing model on test dataset...")

model.eval()  # Set to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No gradients needed for testing
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"\n Test Accuracy: {test_accuracy:.2f}%")
print(f" Correctly classified: {correct}/{total} images")

# ============================================
#  STEP 9 - Save the Model
# ============================================
print("\n Saving model...")

# Save the entire model
torch.save(model, 'mnist_model_complete.pth')
print(" Model saved as 'mnist_model_complete.pth'")

# Save just the state_dict (recommended)
torch.save(model.state_dict(), 'mnist_model_state_dict.pth')
print(" Model state dict saved as 'mnist_model_state_dict.pth'")

# Save checkpoint with more info
checkpoint = {
    'epochs': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_accuracy': test_accuracy,
    'model_architecture': str(model)
}
torch.save(checkpoint, 'mnist_model_checkpoint.pth')
print(" Checkpoint saved as 'mnist_model_checkpoint.pth'")

# ============================================
#  STEP 10 - Quick Demo: Predict a single image
# ============================================
print("\n Quick prediction demo...")

model.eval()
with torch.no_grad():
    # Get one batch from test set
    images, labels = next(iter(test_loader))
    
    # Take first image
    single_image = images[0].unsqueeze(0).to(device)  # Add batch dimension
    true_label = labels[0].item()
    
    # Predict
    output = model(single_image)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_label = torch.argmax(output, dim=1).item()
    confidence = probabilities[0][predicted_label].item()
    
    print(f"   True label: {true_label}")
    print(f"   Predicted: {predicted_label} (confidence: {confidence:.2%})")

print("\n All done! Your MNIST classifier is ready! ")
