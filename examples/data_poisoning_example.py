#!/usr/bin/env python3
"""
Example of using the AI Attack Simulator for data poisoning attacks
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.data_poisoning import DataPoisoningAttack
from utils import save_attack_results, generate_report, visualize_attack_results

def main():
    # Generate a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                              n_redundant=5, random_state=42)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a clean model
    clean_model = LogisticRegression(random_state=42)
    clean_model.fit(X_train, y_train)
    clean_preds = clean_model.predict(X_test)
    clean_accuracy = accuracy_score(y_test, clean_preds)
    print(f"Clean model accuracy: {clean_accuracy:.4f}")
    
    # Create a data poisoning attack
    attack = DataPoisoningAttack(random_seed=42)
    
    # Run the attack with different poisoning methods
    attack_types = ["label_flipping", "backdoor", "clean_label"]
    
    for attack_type in attack_types:
        print(f"\nRunning {attack_type} attack...")
        
        # Set attack parameters
        params = {"percentage": 0.2}  # Poison 20% of the data
        
        # Run the attack
        (poisoned_X_train, poisoned_y_train), attack_results = attack.run(
            X_train, y_train, attack_type=attack_type, params=params
        )
        
        # Train a model on the poisoned data
        poisoned_model = LogisticRegression(random_state=42)
        poisoned_model.fit(poisoned_X_train, poisoned_y_train)
        poisoned_preds = poisoned_model.predict(X_test)
        poisoned_accuracy = accuracy_score(y_test, poisoned_preds)
        
        # Update results with accuracy impact
        attack_results["clean_model_accuracy"] = clean_accuracy
        attack_results["poisoned_model_accuracy"] = poisoned_accuracy
        attack_results["accuracy_degradation"] = clean_accuracy - poisoned_accuracy
        
        print(f"  Attack status: {attack_results['status']}")
        print(f"  Poisoned {attack_results['poisoned_samples']} samples ({attack_results['poisoning_percentage']:.2%})")
        print(f"  Clean model accuracy: {clean_accuracy:.4f}")
        print(f"  Poisoned model accuracy: {poisoned_accuracy:.4f}")
        print(f"  Accuracy degradation: {attack_results['accuracy_degradation']:.4f}")
        
        # Save attack results
        output_dir = "attack_results"
        results_file = save_attack_results(attack_results, output_dir, f"data_poisoning_{attack_type}")
        print(f"  Results saved to {results_file}")
        
        # Generate a report
        report_file = generate_report(attack_results, output_dir, f"data_poisoning_{attack_type}")
        print(f"  Report saved to {report_file}")
        
        # Visualize results
        viz_file = visualize_attack_results(attack_results, output_dir, f"data_poisoning_{attack_type}")
        print(f"  Visualization saved to {viz_file}")

if __name__ == "__main__":
    main()
