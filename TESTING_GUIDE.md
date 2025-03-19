# Kibuka Testing Guide

This guide provides detailed instructions on how to use Kibuka for testing AI systems against various attack types.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Attack Types and Parameters](#attack-types-and-parameters)
3. [Command Line Usage](#command-line-usage)
4. [Programmatic Usage](#programmatic-usage)
5. [Interpreting Results](#interpreting-results)
6. [Real-world Testing Workflow](#real-world-testing-workflow)
7. [Advanced Configuration](#advanced-configuration)

## Prerequisites

Before using Kibuka, ensure you have:

- Python 3.8+
- Required dependencies installed (`pip install -r requirements.txt`)
- A trained ML model to test (supported formats: .pkl, .h5, .pt, or custom loading)
- Test data for certain attack types

## Attack Types and Parameters

### Data Poisoning

Tests how vulnerable your model training process is to contaminated data.

**Parameters:**
- `type`: Poisoning method (`label_flipping`, `backdoor`, `clean_label`)
- `percentage`: Percentage of training data to poison (0.0-1.0)
- `target_classes`: List of classes to target for poisoning
- `trigger_pattern`: For backdoor attacks, the pattern to use as a trigger

**Example:**
```bash
python3 kibuka.py --attack data_poisoning --model path/to/model.pkl --params '{"type": "label_flipping", "percentage": 0.1, "target_classes": [0, 1]}'
```

### Model Inversion

Attempts to reconstruct training data from model outputs.

**Parameters:**
- `attack_type`: Inversion method (`gradient_descent`, `confidence_score`)
- `target_class`: The class to reconstruct samples for
- `num_iterations`: For gradient-based methods, number of iterations
- `max_queries`: For confidence-based methods, maximum queries to make
- `input_shape`: Shape of the input data (e.g., image dimensions)

**Example:**
```bash
python3 kibuka.py --attack model_inversion --model path/to/model.pkl --params '{"attack_type": "gradient_descent", "target_class": 0, "num_iterations": 1000, "input_shape": [28, 28]}'
```

### Evasion Attack

Generates adversarial examples to cause misclassification.

**Parameters:**
- `method`: Attack method (`fgsm`, `pgd`, `carlini_wagner`)
- `epsilon`: Perturbation magnitude
- `norm`: Distance norm to use (`l1`, `l2`, `linf`)
- `targeted`: Whether to target a specific class (boolean)
- `target_class`: If targeted, the class to target

**Example:**
```bash
python3 kibuka.py --attack evasion --model path/to/model.pkl --params '{"method": "fgsm", "epsilon": 0.1, "norm": "linf"}'
```

### Membership Inference

Determines if specific data points were used to train the model.

**Parameters:**
- `attack_type`: Inference method (`confidence_thresholding`, `shadow_model`)
- `threshold`: Confidence threshold for classification
- `num_shadow_models`: For shadow model attacks, number of models to train

**Example:**
```bash
python3 kibuka.py --attack membership_inference --model path/to/model.pkl --params '{"attack_type": "confidence_thresholding", "threshold": 0.8}'
```

### Model Stealing

Attempts to create a copy of the target model by querying it.

**Parameters:**
- `attack_type`: Stealing method (`query_based`, `active_learning`)
- `query_budget`: Maximum number of queries to make
- `substitute_model_type`: Type of model to use as substitute

**Example:**
```bash
python3 kibuka.py --attack model_stealing --model path/to/model.pkl --params '{"attack_type": "query_based", "query_budget": 5000}'
```

## Command Line Usage

### Basic Usage

```bash
# List available attacks
python3 kibuka.py --list-attacks

# Run an attack with default parameters
python3 kibuka.py --attack <attack_type> --model <model_path>

# Run an attack with custom parameters
python3 kibuka.py --attack <attack_type> --model <model_path> --params '<json_params>'

# Specify output directory
python3 kibuka.py --attack <attack_type> --model <model_path> --output <output_dir>

# Use a configuration file
python3 kibuka.py --attack <attack_type> --model <model_path> --config <config_path>
```

### Example Commands

```bash
# Data poisoning with label flipping
python3 kibuka.py --attack data_poisoning --model models/classifier.pkl --params '{"type": "label_flipping", "percentage": 0.15}'

# Model inversion targeting class 3
python3 kibuka.py --attack model_inversion --model models/face_recognition.h5 --params '{"attack_type": "gradient_descent", "target_class": 3}'

# FGSM evasion attack
python3 kibuka.py --attack evasion --model models/malware_detector.pkl --params '{"method": "fgsm", "epsilon": 0.2}'
```

## Programmatic Usage

Kibuka can be imported and used directly in your Python code:

```python
from kibuka import KibukaSimulator
import pickle

# Load your model
with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the simulator
simulator = KibukaSimulator()

# Run an attack
result = simulator.run_attack(
    attack_type="data_poisoning",
    target_model=model,
    attack_params={
        "type": "label_flipping",
        "percentage": 0.1,
        "target_classes": [0, 1]
    }
)

# Process the results
if result['status'] == 'success':
    print(f"Attack succeeded with impact: {result['details']['impact']}")
    # Implement mitigation strategies
else:
    print(f"Attack failed: {result['details'].get('reason', 'Unknown reason')}")
```

## Interpreting Results

Kibuka returns results in a standardized JSON format:

```json
{
  "attack_type": "data_poisoning",
  "subtype": "label_flipping",
  "status": "success",
  "details": {
    "poisoned_percentage": 0.1,
    "technique": "MITRE ATLAS T0020",
    "impact": "Model accuracy degraded by estimated 15-30%"
  }
}
```

Key fields to examine:
- `status`: Whether the attack succeeded
- `details.impact`: Estimated impact on model performance
- `details.technique`: The MITRE ATLAS technique ID

## Real-world Testing Workflow

1. **Baseline Assessment**
   - Train your model on clean data
   - Evaluate baseline performance metrics
   - Document the model architecture and training process

2. **Vulnerability Testing**
   - Run Kibuka attacks against your model
   - Start with default parameters, then customize
   - Test multiple attack vectors

3. **Impact Analysis**
   - Measure performance degradation
   - Identify most successful attack vectors
   - Determine critical vulnerabilities

4. **Mitigation Implementation**
   - Apply defensive techniques based on findings
   - Options include: adversarial training, input sanitization, model distillation

5. **Verification Testing**
   - Re-run attacks on hardened model
   - Verify improvement in resilience
   - Document remaining vulnerabilities

## Advanced Configuration

### Configuration File

You can create a JSON configuration file to set default parameters:

```json
{
  "output_dir": "attack_results",
  "verbosity": "info",
  "max_iterations": 500,
  "random_seed": 42,
  "default_params": {
    "data_poisoning": {
      "type": "label_flipping",
      "percentage": 0.1
    },
    "model_inversion": {
      "attack_type": "gradient_descent",
      "num_iterations": 2000
    }
  }
}
```

Use with:
```bash
python3 kibuka.py --config my_config.json --attack data_poisoning --model my_model.pkl
```

### Custom Attack Implementations

You can extend Kibuka with custom attack implementations:

1. Create a new module in the `attacks/` directory
2. Implement your attack class with a `run()` method
3. Register your attack in `kibuka.py`

See the existing attack implementations for examples.

## Automotive Security Testing

When testing automotive components, consider these specific scenarios:

1. **ECU Testing**
   - Test firmware against model stealing to prevent unauthorized duplication
   - Use evasion attacks to test robustness of control algorithms

2. **Sensor Data Processing**
   - Apply data poisoning to test sensor fusion algorithms
   - Use membership inference to test privacy of collected telemetry

3. **Gateway Security**
   - Test boundary systems with model inversion to assess data leakage risk
   - Apply evasion attacks to test intrusion detection systems

Example command for testing an automotive component:
```bash
python3 kibuka.py --attack evasion --model automotive/sensor_fusion.pkl --params '{"method": "pgd", "epsilon": 0.05, "iterations": 40}'
```
