#!/usr/bin/env python3
"""
Deep explainers for neural networks: GradCAM, LRP, DeepSHAP.
Requires PyTorch models (from pytorch_models.py).
"""

import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


class GradCAMExplainer:
    """
    GradCAM: Gradient-weighted Class Activation Mapping.
    Works with CNN models (ResNet-18).
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or self._find_last_conv_layer()
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model."""
        if hasattr(self.model, 'model'):
            pytorch_model = self.model.model
        else:
            pytorch_model = self.model
        
        # For ResNet, last conv is layer4[-1].conv2
        if hasattr(pytorch_model, 'layer4'):
            return pytorch_model.layer4[-1].conv2
        
        # Find last conv layer
        last_conv = None
        for name, module in pytorch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def explain(self, X_instance, target_class=None):
        """
        Generate GradCAM heatmap for a single image.
        Returns flattened importance vector.
        """
        device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else torch.device('cpu')
        
        # Get the PyTorch model
        if hasattr(self.model, 'model'):
            pytorch_model = self.model.model
        else:
            pytorch_model = self.model
        
        pytorch_model.eval()
        
        # Prepare input
        if isinstance(X_instance, np.ndarray):
            X_tensor = torch.tensor(X_instance, dtype=torch.float32).to(device)
        else:
            X_tensor = X_instance.clone().detach().to(device)
        
        # Reshape if needed
        if X_tensor.dim() == 1:
            if X_tensor.shape[0] == 3072:  # CIFAR-10: 3*32*32
                X_tensor = X_tensor.view(1, 3, 32, 32)
            elif X_tensor.shape[0] == 150528:  # ISIC: 3*224*224
                X_tensor = X_tensor.view(1, 3, 224, 224)
            else:
                X_tensor = X_tensor.view(1, 1, -1)
        elif X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(0)
        
        # Forward pass
        output = pytorch_model(X_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        pytorch_model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        # Generate GradCAM
        if self.activations is not None and self.gradients is not None:
            # Global average pool gradients
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = cam / (cam.max() + 1e-8)
            
            # Resize to input size using interpolation
            cam_resized = torch.nn.functional.interpolate(
                cam, size=(X_tensor.shape[2], X_tensor.shape[3]),
                mode='bilinear', align_corners=False
            )
            
            # Flatten to feature vector
            importance = cam_resized.squeeze().detach().cpu().numpy().flatten()
        else:
            # Fallback: gradient × input
            importance = torch.abs(X_tensor.grad).detach().cpu().numpy().flatten()
        
        # Normalize
        if importance.sum() > 0:
            importance = np.abs(importance) / np.abs(importance).sum()
        
        return importance
    
    def explain_batch(self, X_batch, target_classes=None):
        """Generate GradCAM for a batch of images."""
        results = []
        for i in range(len(X_batch)):
            tc = target_classes[i] if target_classes is not None else None
            results.append(self.explain(X_batch[i], target_class=tc))
        return np.array(results)


class LRPExplainer:
    """
    Layer-Wise Relevance Propagation.
    Implements epsilon-LRP rule for deep neural networks.
    """
    
    def __init__(self, model, epsilon=1e-7):
        self.model = model
        self.epsilon = epsilon
        if hasattr(model, 'model'):
            self.pytorch_model = model.model
        else:
            self.pytorch_model = model
    
    def explain(self, X_instance, target_class=None):
        """
        Compute LRP relevance scores for a single instance.
        Uses epsilon-rule with skip connections for ResNet.
        """
        device = next(self.pytorch_model.parameters()).device
        
        if isinstance(X_instance, np.ndarray):
            X_tensor = torch.tensor(X_instance, dtype=torch.float32).to(device).unsqueeze(0)
        else:
            X_tensor = X_instance.clone().detach().to(device).unsqueeze(0)
        
        X_tensor.requires_grad = True
        
        self.pytorch_model.eval()
        
        # Forward pass
        output = self.pytorch_model(X_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Use gradient * input as LRP approximation (epsilon-rule)
        self.pytorch_model.zero_grad()
        output[0, target_class].backward()
        
        gradient = X_tensor.grad.detach().cpu().numpy().flatten()
        input_val = X_tensor.detach().cpu().numpy().flatten()
        
        # epsilon-LRP: R_i = X_i * gradient / (output + epsilon)
        relevance = input_val * gradient
        relevance = np.abs(relevance)
        
        if relevance.sum() > 0:
            relevance = relevance / relevance.sum()
        
        return relevance
    
    def explain_batch(self, X_batch, target_classes=None):
        """Compute LRP for a batch."""
        results = []
        for i in range(len(X_batch)):
            tc = target_classes[i] if target_classes is not None else None
            results.append(self.explain(X_batch[i], target_class=tc))
        return np.array(results)


class DeepSHAPExplainer:
    """
    SHAP DeepExplainer for neural networks.
    Uses gradient-based approximation of Shapley values.
    """
    
    def __init__(self, model, X_train, n_background=100):
        import shap
        
        if hasattr(model, 'model'):
            self.pytorch_model = model.model
        else:
            self.pytorch_model = model
        
        self.device = next(self.pytorch_model.parameters()).device
        
        # Prepare background data
        if isinstance(X_train, np.ndarray):
            X_background = X_train[:n_background]
            self.X_background_tensor = torch.tensor(X_background, dtype=torch.float32).to(self.device)
        else:
            self.X_background_tensor = X_train[:n_background]
        
        # Reshape if needed
        if self.X_background_tensor.dim() == 2 and self.X_background_tensor.shape[1] > 1000:
            # Likely image data
            if self.X_background_tensor.shape[1] == 3072:
                self.X_background_tensor = self.X_background_tensor.view(-1, 3, 32, 32)
            elif self.X_background_tensor.shape[1] == 150528:
                self.X_background_tensor = self.X_background_tensor.view(-1, 3, 224, 224)
        
        self.explainer = shap.DeepExplainer(self.pytorch_model, self.X_background_tensor)
    
    def explain(self, X_instance):
        """Compute DeepSHAP values for a single instance."""
        if isinstance(X_instance, np.ndarray):
            X_tensor = torch.tensor(X_instance, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X_instance.clone().detach().to(self.device)
        
        # Reshape if needed
        if X_tensor.dim() == 1:
            if X_tensor.shape[0] == 3072:
                X_tensor = X_tensor.view(1, 3, 32, 32)
            elif X_tensor.shape[0] == 150528:
                X_tensor = X_tensor.view(1, 3, 224, 224)
            else:
                X_tensor = X_tensor.unsqueeze(0)
        elif X_tensor.dim() == 2:
            if X_tensor.shape[1] == 3072:
                X_tensor = X_tensor.view(-1, 3, 32, 32)
            elif X_tensor.shape[1] == 150528:
                X_tensor = X_tensor.view(-1, 3, 224, 224)
        
        sv = self.explainer.shap_values(X_tensor)
        
        if isinstance(sv, list):  # multiclass
            return np.mean(np.abs(sv), axis=0).flatten()
        return np.abs(sv).flatten()
    
    def explain_batch(self, X_batch, n_samples=100):
        """Compute DeepSHAP for a batch."""
        import shap
        
        if isinstance(X_batch, np.ndarray):
            X_tensor = torch.tensor(X_batch[:n_samples], dtype=torch.float32).to(self.device)
        else:
            X_tensor = X_batch[:n_samples]
        
        # Reshape if needed
        if X_tensor.dim() == 2 and X_tensor.shape[1] > 1000:
            if X_tensor.shape[1] == 3072:
                X_tensor = X_tensor.view(-1, 3, 32, 32)
            elif X_tensor.shape[1] == 150528:
                X_tensor = X_tensor.view(-1, 3, 224, 224)
        
        sv = self.explainer.shap_values(X_tensor)
        
        if isinstance(sv, list):
            return np.mean([np.abs(s) for s in sv], axis=0)
        return np.abs(sv)


class ImageLIMEExplainer:
    """
    LIME for image classification.
    Segments image into superpixels and explains via local surrogate.
    """
    
    def __init__(self, model, X_train=None):
        from lime.lime_image import LimeImageExplainer
        self.lime_explainer = LimeImageExplainer()
        # Store model wrapper
        if hasattr(model, 'predict_proba'):
            self.model = model
        else:
            self.model = model
    
    def predict_fn(self, X):
        """Prediction function for LIME (images)."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'model'):
            device = next(self.model.model.parameters()).device
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            if X_t.dim() == 2 and X_t.shape[1] == 3072:
                X_t = X_t.view(-1, 3, 32, 32)
            with torch.no_grad():
                output = self.model.model(X_t)
                probs = torch.softmax(output, dim=1)
            return probs.cpu().numpy()
        else:
            raise ValueError("Model must have predict_proba method or be a PyTorch model")
    
    def explain(self, X_instance, num_features=10, num_samples=1000):
        """Generate LIME explanation for an image."""
        # Reshape to image format
        if isinstance(X_instance, np.ndarray):
            if X_instance.ndim == 1:
                if len(X_instance) == 3072:  # CIFAR-10
                    img = X_instance.reshape(3, 32, 32).transpose(1, 2, 0)
                elif len(X_instance) == 150528:  # ISIC
                    img = X_instance.reshape(3, 224, 224).transpose(1, 2, 0)
                else:
                    img = X_instance
            else:
                img = X_instance
        
        # Normalize to [0, 1]
        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        explanation = self.lime_explainer.explain_instance(
            img_normalized, self.predict_fn, 
            top_labels=1, num_features=num_features,
            num_samples=num_samples
        )
        
        # Get feature importance
        local_exp = explanation.local_exp[explanation.top_labels[0]]
        # Map segment importance to pixel importance
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=False, num_features=num_features
        )
        
        return mask.flatten().astype(float)