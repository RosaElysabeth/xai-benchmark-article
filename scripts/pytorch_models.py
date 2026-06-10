#!/usr/bin/env python3
"""
PyTorch models for image and time-series data.
- ResNet-18 (CIFAR-10, ISIC 2019)
- LSTM (PhysioNet, Electricity)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
MODELS_DIR = Path(__file__).parent.parent / "models"


# ========================================================================
# RESNET-18 FOR IMAGE CLASSIFICATION
# ========================================================================

def get_resnet18(num_classes=10, pretrained=True):
    """Load ResNet-18 model, optionally pretrained on ImageNet."""
    import torchvision.models as models
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    
    # Modify final layer for target number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def train_resnet18(X_train, y_train, X_test, y_test, num_classes=10, 
                    epochs=20, batch_size=64, lr=0.001, save_path=None):
    """Train ResNet-18 on image data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Training ResNet-18 on {device}...")
    
    model = get_resnet18(num_classes=num_classes, pretrained=True).to(device)
    
    # Freeze early layers (transfer learning)
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Convert to dataloaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Reshape if flat: (N, C*H*W) → (N, C, H, W)
    if X_train_t.dim() == 2:
        # Assume CIFAR-10: 3x32x32 = 3072
        if X_train_t.shape[1] == 3072:
            X_train_t = X_train_t.view(-1, 3, 32, 32)
            X_test_t = X_test_t.view(-1, 3, 32, 32)
        elif X_train_t.shape[1] == 150528:  # 3*224*224
            X_train_t = X_train_t.view(-1, 3, 224, 224)
            X_test_t = X_test_t.view(-1, 3, 224, 224)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            train_acc = 100 * correct / total
            # Validate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_t)
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = 100 * (val_predicted == y_test_t).sum().item() / len(y_test_t)
            print(f"      Epoch {epoch+1}/{epochs}: loss={running_loss/len(train_loader):.3f}, "
                  f"train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")
    
    # Unfreeze for fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    # Save model
    if save_path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"    Model saved to {save_path}")
    
    return model


class ResNet18Wrapper:
    """Wrapper to make ResNet-18 compatible with scikit-learn-style API."""
    
    def __init__(self, num_classes=10, pretrained=True):
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=20, batch_size=64):
        self.model = train_resnet18(
            X_train, y_train, 
            X_test if X_test is not None else X_train[:100],
            y_test if y_test is not None else y_train[:100],
            num_classes=self.num_classes, epochs=epochs, batch_size=batch_size
        )
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            if X_t.dim() == 2:
                if X_t.shape[1] == 3072:
                    X_t = X_t.view(-1, 3, 32, 32)
                elif X_t.shape[1] == 150528:
                    X_t = X_t.view(-1, 3, 224, 224)
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            if X_t.dim() == 2:
                if X_t.shape[1] == 3072:
                    X_t = X_t.view(-1, 3, 32, 32)
                elif X_t.shape[1] == 150528:
                    X_t = X_t.view(-1, 3, 224, 224)
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()


# ========================================================================
# LSTM FOR TIME-SERIES CLASSIFICATION
# ========================================================================

class LSTMClassifier(nn.Module):
    """LSTM classifier for time-series data."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size) or (batch, features)
        if x.dim() == 2:
            # Treat features as a sequence of length 1
            x = x.unsqueeze(1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return out


class LSTMWrapper:
    """Wrapper to make LSTM compatible with scikit-learn-style API."""
    
    def __init__(self, input_size, hidden_size=64, num_classes=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32, lr=0.001):
        print(f"    Training LSTM on {self.device}...")
        
        self.model = LSTMClassifier(
            self.input_size, self.hidden_size, num_layers=2, 
            num_classes=self.num_classes
        ).to(self.device)
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_t)
                    _, val_pred = torch.max(val_outputs, 1)
                    acc = (val_pred == y_t).float().mean().item() * 100
                print(f"      Epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.3f}, acc={acc:.1f}%")
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()


# ========================================================================
# MLP FOR TIME-SERIES (PyTorch version for DeepSHAP)
# ========================================================================

class MLP(nn.Module):
    """Simple MLP for tabular/time-series data (PyTorch, for DeepSHAP compatibility)."""
    
    def __init__(self, input_size, hidden_sizes=(128, 64, 32), num_classes=2, dropout=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPWrapper:
    """PyTorch MLP wrapper compatible with DeepSHAP."""
    
    def __init__(self, input_size, hidden_sizes=(128, 64, 32), num_classes=2):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def fit(self, X_train, y_train, epochs=100, batch_size=64, lr=0.001):
        self.model = MLP(self.input_size, self.hidden_sizes, self.num_classes).to(self.device)
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        self.model.eval()
        return self
    
    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()


# ========================================================================
# MODEL FACTORY
# ========================================================================

def create_model(model_name, X_train, y_train, task_type="classification"):
    """Factory function to create and train a model."""
    is_classification = task_type == "classification"
    num_classes = len(np.unique(y_train)) if is_classification else None
    input_size = X_train.shape[1]
    
    if model_name == "cnn_resnet18":
        wrapper = ResNet18Wrapper(num_classes=num_classes)
        wrapper.fit(X_train, y_train, epochs=20)
        return wrapper
    
    elif model_name == "lstm":
        wrapper = LSTMWrapper(input_size=input_size, num_classes=num_classes)
        wrapper.fit(X_train, y_train, epochs=50)
        return wrapper
    
    elif model_name == "mlp_pytorch":
        wrapper = MLPWrapper(input_size=input_size, num_classes=num_classes)
        wrapper.fit(X_train, y_train, epochs=100)
        return wrapper
    
    else:
        raise ValueError(f"Unknown PyTorch model: {model_name}")


MODEL_REGISTRY = {
    "cnn_resnet18": {"type": "image", "class": ResNet18Wrapper},
    "lstm": {"type": "timeseries", "class": LSTMWrapper},
    "mlp_pytorch": {"type": "tabular", "class": MLPWrapper},
}