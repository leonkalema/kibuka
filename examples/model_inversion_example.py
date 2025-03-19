#!/usr/bin/env python3
"""
Example of using Kibuka for model inversion attacks
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.model_inversion import ModelInversionAttack
from utils import save_attack_results, generate_report, visualize_attack_results

def main():
    # Load a dataset with visual features (digits)
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize the data
    X = X / 16.0
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    
    # Create a model inversion attack
    attack = ModelInversionAttack(random_seed=42)
    
    # Run the attack with different methods
    attack_types = ["gradient_descent", "confidence_score"]
    
    for attack_type in attack_types:
        print(f"\nRunning {attack_type} attack...")
        
        # Set attack parameters
        params = {
            "input_shape": (8, 8),  # Shape of digit images
            "num_iterations": 1000 if attack_type == "gradient_descent" else None,
            "max_queries": 1000 if attack_type == "confidence_score" else None
        }
        
        # Run the attack for each digit class
        for target_class in range(10):  # 10 digits (0-9)
            print(f"  Targeting class {target_class}...")
            
            # Run the attack
            reconstructed_image, attack_results = attack.run(
                model, target_class=target_class, attack_type=attack_type, params=params
            )
            
            # Save the reconstructed image
            output_dir = "attack_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot the original vs reconstructed image
            plt.figure(figsize=(10, 5))
            
            # Find an example of the target class from the training set
            target_examples = X_train[y_train == target_class]
            if len(target_examples) > 0:
                original_image = target_examples[0].reshape(8, 8)
                
                plt.subplot(1, 2, 1)
                plt.imshow(original_image, cmap='gray')
                plt.title(f"Original Class {target_class}")
                plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image.reshape(8, 8), cmap='gray')
            plt.title(f"Reconstructed Class {target_class}")
            plt.axis('off')
            
            # Save the figure
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_file = os.path.join(output_dir, f"kibuka_model_inversion_{attack_type}_class{target_class}_{timestamp}.png")
            plt.savefig(viz_file)
            plt.close()
            
            # Save attack results
            results_file = save_attack_results(
                attack_results, output_dir, f"kibuka_model_inversion_{attack_type}_class{target_class}"
            )
            
            # Generate a report
            report_file = generate_report(
                attack_results, output_dir, f"kibuka_model_inversion_{attack_type}_class{target_class}"
            )
            
            print(f"    Attack status: {attack_results['status']}")
            print(f"    Confidence: {attack_results.get('confidence', attack_results.get('achieved_confidence', 'N/A'))}")
            print(f"    Results saved to {results_file}")
            print(f"    Report saved to {report_file}")
            print(f"    Visualization saved to {viz_file}")
            
            # Only do one class for demonstration purposes
            break

if __name__ == "__main__":
    main()
