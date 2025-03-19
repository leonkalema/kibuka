"""
Data Poisoning Attack Implementation
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

class DataPoisoningAttack:
    """
    Implements various data poisoning attacks based on NIST and MITRE ATLAS frameworks.
    
    Data poisoning attacks target the training phase of ML models by injecting
    malicious samples that cause the model to learn incorrect patterns.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize the attack with optional random seed for reproducibility."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.attack_types = {
            "label_flipping": self._label_flipping_attack,
            "backdoor": self._backdoor_attack,
            "clean_label": self._clean_label_attack,
        }
    
    def run(self, training_data: Any, labels: Any = None, 
            attack_type: str = "label_flipping", 
            params: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Run the specified data poisoning attack on the training data.
        
        Args:
            training_data: The training data to poison
            labels: The labels associated with the training data
            attack_type: Type of poisoning attack to perform
            params: Additional parameters for the attack
            
        Returns:
            Tuple of (poisoned_data, attack_results)
        """
        if attack_type not in self.attack_types:
            raise ValueError(f"Unknown attack type: {attack_type}. Available types: {list(self.attack_types.keys())}")
        
        params = params or {}
        
        # Run the specific attack implementation
        poisoned_data, poisoned_labels, attack_results = self.attack_types[attack_type](
            training_data, labels, params
        )
        
        # Add metadata about the attack
        attack_results.update({
            "attack_type": "data_poisoning",
            "subtype": attack_type,
            "mitre_technique": "MITRE ATLAS T0020",
            "random_seed": self.random_seed
        })
        
        return (poisoned_data, poisoned_labels), attack_results
    
    def _label_flipping_attack(self, training_data: Any, labels: Any,
                              params: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Implement label flipping attack where a percentage of labels are flipped.
        
        This is one of the simplest poisoning attacks where the attacker
        flips the labels of a subset of the training data.
        """
        percentage = params.get("percentage", 0.1)
        target_class = params.get("target_class", None)
        
        # Create copies to avoid modifying the original data
        poisoned_data = np.copy(training_data)
        poisoned_labels = np.copy(labels)
        
        # Determine which samples to poison
        num_samples = len(training_data)
        num_to_poison = int(num_samples * percentage)
        
        if target_class is not None:
            # Only flip labels for a specific class
            target_indices = np.where(labels == target_class)[0]
            if len(target_indices) > 0:
                poison_indices = np.random.choice(target_indices, 
                                                 min(num_to_poison, len(target_indices)), 
                                                 replace=False)
            else:
                poison_indices = []
        else:
            # Randomly select samples to poison
            poison_indices = np.random.choice(num_samples, num_to_poison, replace=False)
        
        # Flip the labels (in a multi-class scenario, we'd need to decide how to flip)
        # For binary classification, we can simply invert the labels
        if len(poison_indices) > 0:
            if np.unique(labels).shape[0] == 2:  # Binary classification
                poisoned_labels[poison_indices] = 1 - poisoned_labels[poison_indices]
            else:  # Multi-class classification
                # Randomly assign a different class
                unique_labels = np.unique(labels)
                for idx in poison_indices:
                    current_label = poisoned_labels[idx]
                    other_labels = unique_labels[unique_labels != current_label]
                    poisoned_labels[idx] = np.random.choice(other_labels)
        
        attack_results = {
            "poisoned_samples": len(poison_indices),
            "poisoning_percentage": percentage,
            "success_rate": len(poison_indices) / num_samples if num_samples > 0 else 0,
            "status": "success" if len(poison_indices) > 0 else "failed"
        }
        
        return poisoned_data, poisoned_labels, attack_results
    
    def _backdoor_attack(self, training_data: Any, labels: Any,
                        params: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Implement backdoor poisoning attack.
        
        In a backdoor attack, the attacker adds a specific pattern (trigger)
        to some training samples and assigns them a target label. The goal is
        to make the model associate the trigger with the target label.
        """
        percentage = params.get("percentage", 0.05)
        target_label = params.get("target_label", 1)
        trigger_type = params.get("trigger_type", "pattern")
        
        # Create copies to avoid modifying the original data
        poisoned_data = np.copy(training_data)
        poisoned_labels = np.copy(labels)
        
        # Determine which samples to poison
        num_samples = len(training_data)
        num_to_poison = int(num_samples * percentage)
        poison_indices = np.random.choice(num_samples, num_to_poison, replace=False)
        
        # Apply the trigger to the selected samples
        # This is a simplified implementation - in reality, the trigger would
        # depend on the data type (image, text, etc.)
        if trigger_type == "pattern" and len(poisoned_data.shape) > 1:
            # For image data, add a small pattern in the corner
            pattern_size = params.get("pattern_size", 3)
            for idx in poison_indices:
                if len(poisoned_data.shape) == 4:  # Assuming NHWC format for images
                    poisoned_data[idx, :pattern_size, :pattern_size, :] = 1.0
                elif len(poisoned_data.shape) == 3:  # Grayscale or single channel
                    poisoned_data[idx, :pattern_size, :pattern_size] = 1.0
                elif len(poisoned_data.shape) == 2:  # Flattened images
                    feature_dim = poisoned_data.shape[1]
                    img_dim = int(np.sqrt(feature_dim))
                    if img_dim**2 == feature_dim:  # Perfect square
                        for i in range(pattern_size):
                            for j in range(pattern_size):
                                idx_to_modify = i * img_dim + j
                                if idx_to_modify < feature_dim:
                                    poisoned_data[idx, idx_to_modify] = 1.0
        
        # Set the target label for poisoned samples
        poisoned_labels[poison_indices] = target_label
        
        attack_results = {
            "poisoned_samples": len(poison_indices),
            "poisoning_percentage": percentage,
            "target_label": target_label,
            "trigger_type": trigger_type,
            "success_rate": len(poison_indices) / num_samples if num_samples > 0 else 0,
            "status": "success" if len(poison_indices) > 0 else "failed"
        }
        
        return poisoned_data, poisoned_labels, attack_results
    
    def _clean_label_attack(self, training_data: Any, labels: Any,
                           params: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Implement clean-label poisoning attack.
        
        In a clean-label attack, the attacker modifies the features of some samples
        without changing their labels, making the attack harder to detect.
        """
        percentage = params.get("percentage", 0.05)
        target_class = params.get("target_class", 1)
        perturbation_magnitude = params.get("perturbation_magnitude", 0.1)
        
        # Create copies to avoid modifying the original data
        poisoned_data = np.copy(training_data)
        poisoned_labels = np.copy(labels)
        
        # Find samples of the target class
        target_indices = np.where(labels == target_class)[0]
        
        if len(target_indices) == 0:
            return poisoned_data, poisoned_labels, {
                "status": "failed",
                "reason": f"No samples of target class {target_class} found"
            }
        
        # Determine which samples to poison
        num_to_poison = int(len(target_indices) * percentage)
        poison_indices = np.random.choice(target_indices, 
                                         min(num_to_poison, len(target_indices)), 
                                         replace=False)
        
        # Apply subtle perturbations to the selected samples
        # This is a simplified implementation - in reality, the perturbation
        # would be more sophisticated and tailored to the specific model
        for idx in poison_indices:
            # Add random noise
            noise = np.random.normal(0, perturbation_magnitude, size=poisoned_data[idx].shape)
            poisoned_data[idx] += noise
            
            # Ensure the data remains within valid range (assuming [0,1] range)
            poisoned_data[idx] = np.clip(poisoned_data[idx], 0, 1)
        
        attack_results = {
            "poisoned_samples": len(poison_indices),
            "poisoning_percentage": percentage,
            "target_class": target_class,
            "perturbation_magnitude": perturbation_magnitude,
            "success_rate": len(poison_indices) / len(target_indices) if len(target_indices) > 0 else 0,
            "status": "success" if len(poison_indices) > 0 else "failed"
        }
        
        return poisoned_data, poisoned_labels, attack_results
