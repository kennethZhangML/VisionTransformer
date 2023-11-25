import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
import numpy as np

class VisionTransformerExperiments:
    def __init__(self, model_names, train_loader, val_loader, num_epochs = 10, device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_names = model_names
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.results = {}

    def train_and_evaluate(self):
        for model_name in self.model_names:
            print(f"Experimenting with {model_name}")
            model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(self.num_epochs):
                model.train()
                total_loss = 0
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images).logits
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(self.train_loader)

                print(f"Epoch [{epoch+1}/{self.num_epochs}] - Loss: {avg_loss:.4f}")

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images).logits
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                print(f"Validation Accuracy: {accuracy:.2f}%")

            self.results[model_name] = accuracy

    def get_best_model(self):
        best_model_name = max(self.results, key = self.results.get)
        return best_model_name, self.results[best_model_name]

