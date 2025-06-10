# MLP AI Project

A multi-layer perceptron implementation in C++ with support for both regression (Boston Housing) and classification (MNIST) tasks.

## Features

- **Command-line interface** with comprehensive argument parsing
- **Two supported tasks:**
  - Boston Housing price prediction (regression)
  - MNIST digit classification
- **Model persistence:** Save and load trained models
- **Continued training:** Resume training from saved models
- **Early stopping** for MNIST training
- **Comprehensive testing** with automated test suite

## Prerequisites

### Required Datasets

The following datasets are required for full functionality. Place them in the `data/` directory:

1. **Boston Housing Dataset** (`data/boston_housing.csv`)
   - Download from: [Kaggle - Boston House Prices](https://www.kaggle.com/datasets/altavish/boston-housing-dataset)
2. **MNIST Dataset** (`data/mnist_train.csv` and `data/mnist_test.csv`)
   - Download from: [Kaggle - MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

### Build Requirements

- C++ compiler with C++17 support (g++)
- Make

## Building

```bash
# Build the project
make

# Clean and rebuild
make re

# Clean object files
make clean

# Full clean (including binary)
make fclean
```

## Usage

### Basic Commands

```bash
# Show help
./mlp --help

# Train Boston Housing model
./mlp --mode boston --train --epochs 100 --save models/boston_model.txt

# Use trained model for prediction
./mlp --mode boston --predict --load models/boston_model.txt

# Train MNIST classifier
./mlp --mode mnist --train --epochs 50 --save models/mnist_model.txt

# Continue training from saved model
./mlp --mode mnist --train --epochs 20 --load models/mnist_model.txt --save models/mnist_improved.txt

# Use custom dataset path
./mlp --mode boston --train --dataset path/to/custom_dataset.csv --save models/custom_model.txt
```

### Command Line Arguments

**Required:**

- `--mode <task>`: Task mode (`mnist` or `boston`)
- `--train` OR `--predict`: Training or prediction mode

**Optional:**

- `--epochs <num>`: Number of training epochs (default: 100)
- `--dataset <path>`: Path to dataset file
- `--load <path>`: Load existing model from file
- `--save <path>`: Save trained model to file
- `--help`, `-h`: Show help message

## Testing

The project includes a comprehensive testing system to verify all functionality.

### Quick Tests (No Datasets Required)

```bash
# Run basic functionality tests
make test-quick

# Check prerequisites
make test-prereq
```

### Full Test Suite

```bash
# Run comprehensive tests (requires datasets)
make test

# Or run the test script directly
./test_mlp.sh
```

### What the Tests Cover

- ✅ Command line argument parsing
- ✅ Help system functionality
- ✅ Error handling (invalid arguments, missing files, etc.)
- ✅ Boston Housing training and prediction
- ✅ MNIST training and prediction
- ✅ Model saving and loading
- ✅ Continued training from saved models
- ✅ Custom dataset paths
- ✅ Pre-trained model compatibility

### Pre-trained Models

The repository includes pre-trained models in the `models/` directory:

- `boston_test.txt` - Boston Housing regression model
- `mnist_test.txt` - MNIST classification model
- Additional models with variations

These models can be used directly for prediction without training:

```bash
# Use pre-trained Boston model
./mlp --mode boston --predict --load models/boston_test.txt

# Use pre-trained MNIST model
./mlp --mode mnist --predict --load models/mnist_test.txt
```

## Project Structure

```
├── Makefile              # Build configuration
├── test_mlp.sh          # Comprehensive test script
├── README.md            # This file
├── src/                 # Source code
│   ├── main.cpp         # Main application with command parsing
│   └── ...              # Implementation files
├── include/             # Header files
├── data/                # Dataset files (download separately)
├── models/              # Saved model files
└── obj/                 # Object files (generated)
```

## Examples

### Training Workflow

```bash
# 1. Train a model
./mlp --mode mnist --train --epochs 50 --save models/my_mnist.txt

# 2. Test the model
./mlp --mode mnist --predict --load models/my_mnist.txt

# 3. Continue training if needed
./mlp --mode mnist --train --epochs 20 --load models/my_mnist.txt --save models/my_mnist_v2.txt
```

### Regression Example

```bash
# Train Boston Housing model
./mlp --mode boston --train --epochs 100 --save models/boston_regression.txt

# Make predictions
./mlp --mode boston --predict --load models/boston_regression.txt
```

## Troubleshooting

### Dataset Issues

If you get dataset-related errors:

1. Ensure datasets are downloaded and placed in the `data/` directory
2. Check file names match exactly: `boston_housing.csv`, `mnist_train.csv`, `mnist_test.csv`
3. Run `make test-prereq` to check prerequisites

### Build Issues

```bash
# Clean and rebuild
make fclean && make

# Check for missing dependencies
make test-prereq
```

### Model Loading Issues

- Ensure the model file path is correct
- Verify the model was trained for the correct task (boston/mnist)
- Check file permissions

## Development

### Adding New Tests

Add new test cases to `test_mlp.sh`:

```bash
# Add after existing tests
echo
print_info "Test N: Your New Test"
if ./mlp --your-test-command > /dev/null 2>&1; then
    print_success "Your test passed"
else
    print_error "Your test failed"
    exit 1
fi
```

### Makefile Targets

- `make` - Build the project
- `make test` - Run full test suite
- `make test-quick` - Run basic tests
- `make test-prereq` - Check prerequisites
- `make clean` - Clean object files
- `make help` - Show all available targets

## License

This project is for educational purposes.
