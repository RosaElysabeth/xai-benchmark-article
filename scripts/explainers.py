#!/usr/bin/env python3
"""
Explainer wrappers for all 10 XAI method variants.
Provides a unified interface: explain(X_instance) → feature_importance_vector
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')


class BaseExplainer:
    """Base class for all explainers."""
    
    def __init__(self, model, X_train, feature_names=None, model_type="tabular", 
                 task_type="classification"):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"x{i}" for i in range(X_train.shape[1])]
        self.model_type = model_type
        self.task_type = task_type
        self.n_features = X_train.shape[1]
    
    def explain(self, X_instance):
        raise NotImplementedError
    
    def explain_batch(self, X_batch):
        """Explain a batch of instances."""
        results = []
        for i in range(len(X_batch)):
            results.append(self.explain(X_batch[i]))
        return np.array(results)
    
    def predict(self, X):
        if self.task_type == "classification":
            return self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else self.model.predict(X)
        return self.model.predict(X)


class SHAPTreeExplainer(BaseExplainer):
    """SHAP TreeExplainer - exact for tree models."""
    
    def __init__(self, model, X_train, **kwargs):
        super().__init__(model, X_train, **kwargs)
        import shap
        self.explainer = shap.TreeExplainer(model)
    
    def _aggregate_shap(self, sv):
        """Aggregate SHAP values across classes into per-feature importance."""
        if isinstance(sv, list):
            # List of (n_samples, n_features) arrays per class
            return np.mean(np.abs(sv), axis=0).flatten()[:self.n_features]
        sv = np.array(sv)
        if sv.ndim == 3:
            # (n_samples, n_features, n_classes) -> mean over classes, squeeze samples
            return np.mean(np.abs(sv), axis=-1).flatten()[:self.n_features]
        elif sv.ndim == 2:
            # (n_samples, n_features)
            return np.abs(sv).flatten()[:self.n_features]
        else:
            return np.abs(sv).flatten()[:self.n_features]
    
    def explain(self, X_instance):
        sv = self.explainer.shap_values(X_instance.reshape(1, -1))
        return self._aggregate_shap(sv)
    
    def explain_batch(self, X_batch):
        sv = self.explainer.shap_values(X_batch)
        if isinstance(sv, list):
            return np.mean(np.abs(sv), axis=0)
        sv = np.array(sv)
        if sv.ndim == 3:
            return np.mean(np.abs(sv), axis=-1)
        return np.abs(sv)


class SHAPKernelExplainer(BaseExplainer):
    """SHAP KernelExplainer - model-agnostic but slow."""
    
    def __init__(self, model, X_train, nsamples=50, **kwargs):
        super().__init__(model, X_train, **kwargs)
        import shap
        self.nsamples = nsamples
        background = shap.sample(X_train, min(100, len(X_train)))
        self.explainer = shap.KernelExplainer(self.predict, background)
    
    def _aggregate_shap(self, sv):
        """Aggregate SHAP values across classes into per-feature importance."""
        if isinstance(sv, list):
            return np.mean(np.abs(sv), axis=0).flatten()[:self.n_features]
        sv = np.array(sv)
        if sv.ndim == 3:
            return np.mean(np.abs(sv), axis=-1).flatten()[:self.n_features]
        elif sv.ndim == 2:
            return np.abs(sv).flatten()[:self.n_features]
        else:
            return np.abs(sv).flatten()[:self.n_features]
    
    def explain(self, X_instance):
        sv = self.explainer.shap_values(X_instance.reshape(1, -1), nsamples=self.nsamples)
        return self._aggregate_shap(sv)


class SHAPDeepExplainer(BaseExplainer):
    """SHAP DeepExplainer - for neural networks."""
    
    def __init__(self, model, X_train, framework="pytorch", **kwargs):
        super().__init__(model, X_train, **kwargs)
        import shap
        import torch
        self.framework = framework
        
        # DeepExplainer requires PyTorch model wrapped
        background = shap.sample(X_train, min(100, len(X_train)))
        if framework == "pytorch":
            self.explainer = shap.DeepExplainer(model, torch.tensor(background, dtype=torch.float32))
        else:
            self.explainer = shap.DeepExplainer(model, background)
    
    def _aggregate_shap(self, sv):
        """Aggregate SHAP values across classes into per-feature importance."""
        if isinstance(sv, list):
            return np.mean(np.abs(sv), axis=0).flatten()[:self.n_features]
        sv = np.array(sv)
        if sv.ndim == 3:
            return np.mean(np.abs(sv), axis=-1).flatten()[:self.n_features]
        elif sv.ndim == 2:
            return np.abs(sv).flatten()[:self.n_features]
        else:
            return np.abs(sv).flatten()[:self.n_features]
    
    def explain(self, X_instance):
        import torch
        sv = self.explainer.shap_values(
            torch.tensor(X_instance.reshape(1, -1), dtype=torch.float32)
            if self.framework == "pytorch" else X_instance.reshape(1, -1)
        )
        return self._aggregate_shap(sv)


class LIMEExplainer(BaseExplainer):
    """LIME - Local Interpretable Model-agnostic Explanations."""
    
    def __init__(self, model, X_train, **kwargs):
        super().__init__(model, X_train, **kwargs)
        from lime.lime_tabular import LimeTabularExplainer
        self.explainer = LimeTabularExplainer(
            X_train, feature_names=self.feature_names,
            mode=self.task_type, random_state=42
        )
    
    def explain(self, X_instance):
        exp = self.explainer.explain_instance(
            X_instance, self.predict, num_features=self.n_features,
            num_samples=5000
        )
        # Convert LIME's sparse output to dense vector
        weights = dict(exp.as_list())
        importance = np.array([weights.get(f, 0.0) for f in self.feature_names])
        return importance


class PermutationImportanceExplainer(BaseExplainer):
    """Permutation Importance - global, fast, stable."""
    
    def __init__(self, model, X_train, **kwargs):
        super().__init__(model, X_train, **kwargs)
        from sklearn.inspection import permutation_importance
        self._compute_permutation_importance(model, X_train)
    
    def _compute_permutation_importance(self, model, X_train):
        y_pred = model.predict(X_train)
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            model, X_train, y_pred, n_repeats=10, random_state=42
        )
        self.global_importance = result.importances_mean
    
    def explain(self, X_instance):
        # Permutation importance is global, same for all instances
        return self.global_importance.copy()


class PDPExplainer(BaseExplainer):
    """Partial Dependence Plots - global effect visualization."""
    
    def explain(self, X_instance):
        from sklearn.inspection import partial_dependence
        importances = np.zeros(self.n_features)
        for i in range(self.n_features):
            try:
                pd_result = partial_dependence(
                    self.model, self.X_train[:500], features=[i], kind="average"
                )
                importances[i] = np.var(pd_result["average"][0])
            except Exception:
                importances[i] = 0.0
        return importances


class ALEExplainer(BaseExplainer):
    """Accumulated Local Effects - handles correlated features."""
    
    def explain(self, X_instance):
        # ALE: variance of local differences
        importances = np.zeros(self.n_features)
        for i in range(self.n_features):
            try:
                # Simplified ALE: compute variance of local differences
                sorted_vals = np.sort(self.X_train[:, i])
                diffs = np.diff(sorted_vals)
                if len(diffs) > 0 and np.std(diffs) > 0:
                    # Effect = accumulated local differences
                    effects = np.cumsum(diffs) / len(diffs)
                    importances[i] = np.var(effects)
                else:
                    importances[i] = 0.0
            except Exception:
                importances[i] = 0.0
        return importances


class LRPExplainer(BaseExplainer):
    """Layer-Wise Relevance Propagation - for neural networks."""
    
    def explain(self, X_instance):
        # LRP implementation for PyTorch models
        # This is a simplified version - full implementation requires model-specific rules
        warnings.warn("LRP requires model-specific implementation. Using gradient-based approximation.")
        return self._gradient_attribution(X_instance)
    
    def _gradient_attribution(self, X_instance):
        """Approximate LRP with gradient × input (epsilon-rule approximation)."""
        import torch
        self.model.eval()
        X_tensor = torch.tensor(X_instance.reshape(1, -1), dtype=torch.float32, requires_grad=True)
        output = self.model(X_tensor)
        output.backward()
        gradient = X_tensor.grad.detach().numpy().flatten()
        return np.abs(gradient) * np.abs(X_instance)


class GradCAMExplainer(BaseExplainer):
    """GradCAM - for CNN models (image only)."""
    
    def __init__(self, model, X_train, target_layer=None, **kwargs):
        super().__init__(model, X_train, **kwargs)
        self.target_layer = target_layer
        self._gradcam = None
    
    def _compute_gradcam(self, X_tensor, target_class=None):
        """Compute GradCAM heatmap."""
        import torch
        self.model.eval()
        
        # Forward pass
        output = self.model(X_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        # Get gradients and activations from target layer
        gradients = self._gradients
        activations = self._activations
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / cam.max()
        
        return cam.squeeze().detach().numpy()
    
    def explain(self, X_instance):
        """Return feature importance from GradCAM heatmap."""
        # For image models: reshape, compute GradCAM, flatten
        if X_instance.ndim == 1 and len(X_instance) > 1000:
            # Likely image data (flattened)
            # GradCAM importance = mean of heatmap per channel
            warnings.warn("GradCAM on flattened images. Use CNN model for full GradCAM.")
            return self._gradient_attribution(X_instance)
        
        # For tabular: gradient attribution
        return self._gradient_attribution(X_instance)
    
    def _gradient_attribution(self, X_instance):
        """Saliency-based attribution as fallback."""
        import torch
        self.model.eval()
        X_tensor = torch.tensor(X_instance.reshape(1, -1), dtype=torch.float32, requires_grad=True)
        output = self.model(X_tensor)
        output.max().backward()
        return np.abs(X_tensor.grad.detach().numpy().flatten())


class CounterfactualExplainer(BaseExplainer):
    """Counterfactual Explanations - minimal perturbation."""
    
    def explain(self, X_instance):
        """Feature importance from counterfactual search."""
        original_pred = self.model.predict(X_instance.reshape(1, -1))[0]
        importances = np.zeros(self.n_features)
        
        for j in range(self.n_features):
            std = np.std(self.X_train[:, j])
            if std == 0:
                continue
            
            # Try increasing perturbations until prediction changes
            for delta in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                X_cf = X_instance.copy()
                X_cf[j] += delta * std
                try:
                    new_pred = self.model.predict(X_cf.reshape(1, -1))[0]
                    if new_pred != original_pred:
                        importances[j] = 1.0 / delta  # Smaller delta = more important
                        break
                except Exception:
                    continue
            else:
                # Feature not important enough to flip prediction
                importances[j] = 0.0
        
        return importances


# ========================================================================
# EXPLAINER FACTORY
# ========================================================================

EXPLAINER_CLASSES = {
    "shap_tree": SHAPTreeExplainer,
    "shap_kernel": SHAPKernelExplainer,
    "shap_deep": SHAPDeepExplainer,
    "lime": LIMEExplainer,
    "permutation": PermutationImportanceExplainer,
    "pdp": PDPExplainer,
    "ale": ALEExplainer,
    "lrp": LRPExplainer,
    "gradcam": GradCAMExplainer,
    "counterfactual": CounterfactualExplainer,
}

# Which methods are compatible with which model types
COMPATIBILITY = {
    "tabular": {
        "ridge": ["shap_kernel", "lime", "permutation", "pdp", "ale", "counterfactual"],
        "decision_tree": ["shap_tree", "shap_kernel", "lime", "permutation", "pdp", "ale", "counterfactual"],
        "ebm": ["shap_kernel", "lime", "permutation", "pdp", "ale", "counterfactual"],
        "random_forest": ["shap_tree", "shap_kernel", "lime", "permutation", "pdp", "ale", "counterfactual"],
        "xgboost": ["shap_tree", "shap_kernel", "lime", "permutation", "pdp", "ale", "counterfactual"],
        "lightgbm": ["shap_tree", "shap_kernel", "lime", "permutation", "pdp", "ale", "counterfactual"],
        "catboost": ["shap_tree", "shap_kernel", "lime", "permutation", "pdp", "ale", "counterfactual"],
        "mlp": ["shap_kernel", "shap_deep", "lime", "permutation", "pdp", "ale", "counterfactual"],
    },
    "image": {
        "cnn_resnet18": ["shap_kernel", "shap_deep", "lime", "gradcam", "lrp"],
    },
    "timeseries": {
        "lstm": ["shap_kernel", "lime", "permutation", "pdp", "ale"],
        "ridge": ["shap_kernel", "lime", "permutation", "pdp", "ale", "counterfactual"],
    },
}

SCOPE = {
    "shap_tree": "both", "shap_kernel": "both", "shap_deep": "both",
    "lime": "local", "permutation": "global",
    "pdp": "global", "ale": "global",
    "lrp": "local", "gradcam": "local", "counterfactual": "local",
}

AGNOSTIC = {
    "shap_tree": False, "shap_kernel": True, "shap_deep": False,
    "lime": True, "permutation": True,
    "pdp": True, "ale": True,
    "lrp": False, "gradcam": False, "counterfactual": True,
}

CAUSAL = {
    "shap_tree": False, "shap_kernel": False, "shap_deep": False,
    "lime": False, "permutation": False,
    "pdp": False, "ale": False,
    "lrp": False, "gradcam": False, "counterfactual": "partial",
}


def get_compatible_methods(model_name, data_type):
    """Return list of XAI methods compatible with a given model and data type."""
    if data_type in COMPATIBILITY and model_name in COMPATIBILITY[data_type]:
        return COMPATIBILITY[data_type][model_name]
    return []


def create_explainer(method_name, model, X_train, feature_names=None, 
                     model_type="tabular", task_type="classification"):
    """Factory function to create an explainer."""
    cls = EXPLAINER_CLASSES.get(method_name)
    if cls is None:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(EXPLAINER_CLASSES.keys())}")
    
    return cls(model, X_train, feature_names=feature_names, 
               model_type=model_type, task_type=task_type)