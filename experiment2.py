import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

embedding_dims = [128, 256, 512] 
num_heads_list = [4, 8, 16]  
depths = [6, 12]  
learning_rates = [1e-3, 5e-4]  

experiment_results = {}

for embed_dim in embedding_dims:
    for num_heads in num_heads_list:
        for depth in depths:
            for learning_rate in learning_rates:
                model = VisionTransformer(
                    img_size = image_size,
                    patch_size = 16,
                    in_channels = 3,
                    num_classes = num_classes,
                    embed_dim = embed_dim,
                    depth = depth,
                    num_heads = num_heads,
                    mlp_ratio = 2.0,
                    attn_drop_rate = 0.1,
                    drop_rate = 0.1,
                    norm_layer = nn.LayerNorm,
                )

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr = learning_rate)

                num_epochs = 10
                train_losses = []

                for epoch in range(num_epochs):
                    model.train()
                    total_loss = 0.0
                    for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=True):
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels.squeeze())
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                    avg_loss = total_loss / len(dataset)
                    train_losses.append(avg_loss)

                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in DataLoader(dataset, batch_size=batch_size):
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.squeeze()).sum().item()

                    accuracy = 100 * correct / total

                experiment_results[(embed_dim, num_heads, depth, learning_rate)] = {
                    'train_losses': train_losses,
                    'accuracy': accuracy,
                }

best_accuracy = 0.0
best_config = None

for config, results in experiment_results.items():
    embed_dim, num_heads, depth, learning_rate = config
    accuracy = results['accuracy']
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_config = config

print(f"Best configuration: Embedding Dim = {best_config[0]}, "
      f"Num Heads = {best_config[1]}, Depth = {best_config[2]}, Learning Rate = {best_config[3]}")
print(f"Best Accuracy: {best_accuracy:.2f}%")

best_train_losses = experiment_results[best_config]['train_losses']
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
plt.plot(best_train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs (Best Configuration)")
plt.legend()
plt.show()
