import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import seaborn as sns 

from ViT import VisionTransformer
torch.manual_seed(42)

class SyntheticColorDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, image_size, num_classes):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = torch.randint(0, self.num_classes, (1,))
        
        image = torch.zeros((3, self.image_size, self.image_size))
        color = torch.randint(0, 256, (3,))  
        image[:, :, :] = color.view(3, 1, 1)
        
        return image, label

batch_size = 16
image_size = 64  
num_classes = 4  

dataset = SyntheticColorDataset(num_samples = 1000, image_size = image_size, num_classes = num_classes)
data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

model = VisionTransformer(
    img_size = image_size,
    patch_size = 16,
    in_channels = 3,
    num_classes = num_classes,
    embed_dim = 256,
    depth = 6,
    num_heads = 4,
    mlp_ratio = 2.0,
    attn_drop_rate = 0.1,
    drop_rate = 0.1,
    norm_layer = nn.LayerNorm,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

num_epochs = 10
train_losses = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test dataset: {accuracy:.2f}%")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test dataset: {accuracy:.2f}%")

plt.figure(figsize = (10, 5))
sns.set_style("whitegrid")
plt.plot(train_losses, label = "Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()