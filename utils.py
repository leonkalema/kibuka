"""
Utility functions for AI Attack Simulator
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union

def save_attack_results(results: Dict[str, Any], output_dir: str, 
                       attack_type: str, timestamp: str = None) -> str:
    """
    Save attack results to a JSON file.
    
    Args:
        results: Attack results dictionary
        output_dir: Directory to save results
        attack_type: Type of attack
        timestamp: Optional timestamp for the filename
        
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if timestamp is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{attack_type}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filepath

def generate_report(results: Dict[str, Any], output_dir: str, 
                   attack_type: str, format: str = "text") -> str:
    """
    Generate a report from attack results.
    
    Args:
        results: Attack results dictionary
        output_dir: Directory to save the report
        attack_type: Type of attack
        format: Report format (text, html, or pdf)
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "text":
        filename = f"{attack_type}_report_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"AI Attack Simulator Report\n")
            f.write(f"========================\n\n")
            f.write(f"Attack Type: {attack_type}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write(f"Results Summary:\n")
            f.write(f"---------------\n")
            
            # Write attack-specific details
            if "status" in results:
                f.write(f"Status: {results['status']}\n")
            
            if "details" in results:
                f.write(f"\nDetails:\n")
                for key, value in results["details"].items():
                    f.write(f"  {key}: {value}\n")
            else:
                for key, value in results.items():
                    if key not in ["status", "attack_type"]:
                        f.write(f"  {key}: {value}\n")
            
            f.write(f"\nMITRE ATT&CK for AI Reference:\n")
            if attack_type == "data_poisoning":
                f.write(f"  Technique: T0020 - Poison Training Data\n")
            elif attack_type == "model_inversion":
                f.write(f"  Technique: T0019 - Exfiltrate Information via ML Inference\n")
            elif attack_type == "evasion":
                f.write(f"  Technique: T0015 - Model Evasion\n")
            elif attack_type == "membership_inference":
                f.write(f"  Technique: T0018 - Infer Training Data\n")
            elif attack_type == "model_stealing":
                f.write(f"  Technique: T0016 - Model Stealing\n")
            
            f.write(f"\nRecommendations:\n")
            f.write(f"---------------\n")
            if attack_type == "data_poisoning":
                f.write(f"  1. Implement data provenance tracking\n")
                f.write(f"  2. Use robust training techniques\n")
                f.write(f"  3. Validate training data for anomalies\n")
            elif attack_type == "model_inversion":
                f.write(f"  1. Limit model output precision\n")
                f.write(f"  2. Add noise to model outputs\n")
                f.write(f"  3. Implement rate limiting for queries\n")
            elif attack_type == "evasion":
                f.write(f"  1. Train with adversarial examples\n")
                f.write(f"  2. Implement input preprocessing\n")
                f.write(f"  3. Use ensemble models\n")
            elif attack_type == "membership_inference":
                f.write(f"  1. Apply differential privacy during training\n")
                f.write(f"  2. Reduce model overfitting\n")
                f.write(f"  3. Limit output confidence scores\n")
            elif attack_type == "model_stealing":
                f.write(f"  1. Implement query rate limiting\n")
                f.write(f"  2. Add noise to model outputs\n")
                f.write(f"  3. Use watermarking techniques\n")
    
    elif format == "html":
        # Simple HTML report implementation
        filename = f"{attack_type}_report_{timestamp}.html"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"<!DOCTYPE html>\n<html>\n<head>\n")
            f.write(f"  <title>AI Attack Simulator Report - {attack_type}</title>\n")
            f.write(f"  <style>\n")
            f.write(f"    body {{ font-family: Arial, sans-serif; margin: 40px; }}\n")
            f.write(f"    h1, h2 {{ color: #2c3e50; }}\n")
            f.write(f"    .container {{ max-width: 800px; margin: 0 auto; }}\n")
            f.write(f"    .section {{ margin-bottom: 20px; }}\n")
            f.write(f"    .status-success {{ color: green; }}\n")
            f.write(f"    .status-failed {{ color: red; }}\n")
            f.write(f"    .status-partial {{ color: orange; }}\n")
            f.write(f"  </style>\n")
            f.write(f"</head>\n<body>\n")
            f.write(f"  <div class='container'>\n")
            f.write(f"    <h1>AI Attack Simulator Report</h1>\n")
            f.write(f"    <div class='section'>\n")
            f.write(f"      <p><strong>Attack Type:</strong> {attack_type}</p>\n")
            f.write(f"      <p><strong>Timestamp:</strong> {timestamp}</p>\n")
            f.write(f"    </div>\n")
            
            # Status section
            status_class = "status-success"
            if "status" in results:
                if results["status"] == "failed":
                    status_class = "status-failed"
                elif results["status"] == "partial_success":
                    status_class = "status-partial"
                
                f.write(f"    <div class='section'>\n")
                f.write(f"      <h2>Results Summary</h2>\n")
                f.write(f"      <p><strong>Status:</strong> <span class='{status_class}'>{results['status']}</span></p>\n")
                
                # Write attack-specific details
                if "details" in results:
                    for key, value in results["details"].items():
                        f.write(f"      <p><strong>{key}:</strong> {value}</p>\n")
                else:
                    for key, value in results.items():
                        if key not in ["status", "attack_type"]:
                            f.write(f"      <p><strong>{key}:</strong> {value}</p>\n")
                
                f.write(f"    </div>\n")
            
            # MITRE reference section
            f.write(f"    <div class='section'>\n")
            f.write(f"      <h2>MITRE ATT&CK for AI Reference</h2>\n")
            if attack_type == "data_poisoning":
                f.write(f"      <p><strong>Technique:</strong> T0020 - Poison Training Data</p>\n")
            elif attack_type == "model_inversion":
                f.write(f"      <p><strong>Technique:</strong> T0019 - Exfiltrate Information via ML Inference</p>\n")
            elif attack_type == "evasion":
                f.write(f"      <p><strong>Technique:</strong> T0015 - Model Evasion</p>\n")
            elif attack_type == "membership_inference":
                f.write(f"      <p><strong>Technique:</strong> T0018 - Infer Training Data</p>\n")
            elif attack_type == "model_stealing":
                f.write(f"      <p><strong>Technique:</strong> T0016 - Model Stealing</p>\n")
            f.write(f"    </div>\n")
            
            # Recommendations section
            f.write(f"    <div class='section'>\n")
            f.write(f"      <h2>Recommendations</h2>\n")
            f.write(f"      <ul>\n")
            if attack_type == "data_poisoning":
                f.write(f"        <li>Implement data provenance tracking</li>\n")
                f.write(f"        <li>Use robust training techniques</li>\n")
                f.write(f"        <li>Validate training data for anomalies</li>\n")
            elif attack_type == "model_inversion":
                f.write(f"        <li>Limit model output precision</li>\n")
                f.write(f"        <li>Add noise to model outputs</li>\n")
                f.write(f"        <li>Implement rate limiting for queries</li>\n")
            elif attack_type == "evasion":
                f.write(f"        <li>Train with adversarial examples</li>\n")
                f.write(f"        <li>Implement input preprocessing</li>\n")
                f.write(f"        <li>Use ensemble models</li>\n")
            elif attack_type == "membership_inference":
                f.write(f"        <li>Apply differential privacy during training</li>\n")
                f.write(f"        <li>Reduce model overfitting</li>\n")
                f.write(f"        <li>Limit output confidence scores</li>\n")
            elif attack_type == "model_stealing":
                f.write(f"        <li>Implement query rate limiting</li>\n")
                f.write(f"        <li>Add noise to model outputs</li>\n")
                f.write(f"        <li>Use watermarking techniques</li>\n")
            f.write(f"      </ul>\n")
            f.write(f"    </div>\n")
            
            f.write(f"  </div>\n")
            f.write(f"</body>\n</html>\n")
    
    return filepath

def visualize_attack_results(results: Dict[str, Any], output_dir: str, 
                            attack_type: str) -> str:
    """
    Visualize attack results.
    
    Args:
        results: Attack results dictionary
        output_dir: Directory to save visualizations
        attack_type: Type of attack
        
    Returns:
        Path to the visualization file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{attack_type}_viz_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.figure(figsize=(10, 6))
    
    if attack_type == "data_poisoning":
        # Visualize data poisoning results
        if "poisoning_percentage" in results:
            plt.bar(["Clean Data", "Poisoned Data"], 
                   [1 - results["poisoning_percentage"], results["poisoning_percentage"]])
            plt.ylabel("Proportion")
            plt.title("Data Poisoning Attack Results")
    
    elif attack_type == "model_inversion":
        # Visualize model inversion results
        if "confidence" in results:
            plt.bar(["Confidence"], [results["confidence"]])
            plt.ylim(0, 1)
            plt.title("Model Inversion Attack Confidence")
    
    elif attack_type == "evasion":
        # Visualize evasion attack results
        if "success_rate" in results:
            plt.bar(["Success Rate"], [results["success_rate"]])
            plt.ylim(0, 1)
            plt.title("Evasion Attack Success Rate")
    
    elif attack_type == "membership_inference":
        # Visualize membership inference results
        if all(k in results for k in ["accuracy", "precision", "recall"]):
            plt.bar(["Accuracy", "Precision", "Recall"], 
                   [results["accuracy"], results["precision"], results["recall"]])
            plt.ylim(0, 1)
            plt.title("Membership Inference Attack Performance")
    
    elif attack_type == "model_stealing":
        # Visualize model stealing results
        if "functional_similarity" in results:
            plt.bar(["Functional Similarity"], [results["functional_similarity"]])
            plt.ylim(0, 1)
            plt.title("Model Stealing Attack - Functional Similarity")
    
    plt.savefig(filepath)
    plt.close()
    
    return filepath
