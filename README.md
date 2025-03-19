# Kibuka - AI Attack Simulator

A modular, lightweight tool for testing AI systems against common attacks based on NIST and MITRE ATT&CK frameworks for AI.

## Features

- **Comprehensive Attack Coverage**: Implements key AI attack techniques from MITRE ATLAS framework
- **Modular Design**: Each attack type is implemented as a separate module
- **Detailed Reporting**: Generates reports with attack results and mitigation recommendations
- **Visualization**: Creates visual representations of attack outcomes
- **Easy Integration**: Simple API for integration with existing AI systems

## Supported Attacks

1. **Data Poisoning** (MITRE ATLAS T0020)
   - Label flipping
   - Backdoor attacks
   - Clean-label poisoning

2. **Model Inversion** (MITRE ATLAS T0019)
   - Gradient-based reconstruction
   - Confidence score exploitation

3. **Evasion Attacks** (MITRE ATLAS T0015)
   - Fast Gradient Sign Method (FGSM)
   - Projected Gradient Descent (PGD)
   - Carlini & Wagner attacks

4. **Membership Inference** (MITRE ATLAS T0018)
   - Confidence thresholding
   - Shadow model attacks

5. **Model Stealing** (MITRE ATLAS T0016)
   - Query-based extraction
   - Active learning approaches

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kibuka.git
cd kibuka

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from kibuka import KibukaSimulator

# Initialize the simulator
simulator = KibukaSimulator()

# List available attacks
attacks = simulator.list_available_attacks()
print(f"Available attacks: {attacks}")

# Run a data poisoning attack
result = simulator.run_attack(
    attack_type="data_poisoning",
    target_model=your_model,  # Your ML model or API endpoint
    attack_params={
        "type": "label_flipping",
        "percentage": 0.1
    }
)

print(f"Attack status: {result['status']}")
print(f"Attack details: {result['details']}")
```

## Command Line Usage

```bash
# List available attacks
python kibuka.py --list-attacks

# Run an attack with basic options
python kibuka.py --attack data_poisoning --model path/to/model.pkl --output results/

# Run an attack with detailed parameters
python kibuka.py --attack data_poisoning --model path/to/model.pkl --params '{"type": "label_flipping", "percentage": 0.1, "target_classes": [0, 1]}'

# Run model inversion attack with custom parameters
python kibuka.py --attack model_inversion --model path/to/model.pkl --params '{"attack_type": "gradient_descent", "target_class": 0, "num_iterations": 1000}'

# Run evasion attack with specific method and epsilon
python kibuka.py --attack evasion --model path/to/model.pkl --params '{"method": "fgsm", "epsilon": 0.1, "norm": "linf"}'
```

For detailed usage instructions and parameter explanations for each attack type, please refer to the [Testing Guide](TESTING_GUIDE.md).

## Examples

Check the `examples/` directory for detailed examples of each attack type:

- `data_poisoning_example.py`: Demonstrates data poisoning attacks
- `model_inversion_example.py`: Shows how to extract training data
- `evasion_example.py`: Generates adversarial examples
- `membership_inference_example.py`: Tests privacy leakage
- `model_stealing_example.py`: Demonstrates model extraction



## Security Notice

This tool is intended for defensive security testing and educational purposes only. Always obtain proper authorization before testing AI systems.

## License

MIT

## Requirements

- Python 3.8+
- numpy>=1.20.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- tensorflow>=2.8.0 (optional)
- torch>=1.10.0 (optional)
- matplotlib>=3.5.0
- click>=8.0.0
