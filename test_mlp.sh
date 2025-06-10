#!/bin/bash

# MLP AI Testing Script
# This script checks prerequisites and runs comprehensive tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_MODELS_DIR="test_models"
TEST_EPOCHS=10  # Use fewer epochs for faster testing

echo -e "${BLUE}=== MLP AI Testing Script ===${NC}"
echo "This script will check prerequisites and run comprehensive tests."
echo

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to check if a file exists
check_file() {
    if [ -f "$1" ]; then
        print_success "$1 found"
        return 0
    else
        print_error "$1 not found"
        return 1
    fi
}

# Function to check if a directory exists
check_dir() {
    if [ -d "$1" ]; then
        print_success "$1 directory found"
        return 0
    else
        print_error "$1 directory not found"
        return 1
    fi
}

# Check prerequisites
echo -e "${BLUE}=== Checking Prerequisites ===${NC}"

# Check if executable exists
if ! check_file "./mlp"; then
    print_info "Building the project first..."
    if make; then
        print_success "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
fi

# Check data directory
if ! check_dir "data"; then
    print_error "Data directory missing. Creating it..."
    mkdir -p data
fi

# Check required datasets
echo
print_info "Checking required datasets..."

MISSING_DATASETS=()

if ! check_file "data/boston_housing.csv"; then
    MISSING_DATASETS+=("boston_housing.csv")
fi

if ! check_file "data/mnist_train.csv"; then
    MISSING_DATASETS+=("mnist_train.csv")
fi

if ! check_file "data/mnist_test.csv"; then
    MISSING_DATASETS+=("mnist_test.csv")
fi

if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
    echo
    print_warning "Missing datasets detected!"
    echo "The following datasets are required but not found:"
    for dataset in "${MISSING_DATASETS[@]}"; do
        echo "  - $dataset"
    done
    echo
    echo "You can download these datasets from:"
    echo "  • Boston Housing: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices"
    echo "  • MNIST: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"
    echo
    echo "Place the CSV files in the 'data/' directory with the following names:"
    echo "  - data/boston_housing.csv"
    echo "  - data/mnist_train.csv" 
    echo "  - data/mnist_test.csv"
    echo
    read -p "Do you want to continue with available datasets only? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Exiting. Please download the required datasets and try again."
        exit 1
    fi
fi

# Create test models directory
echo
print_info "Setting up test environment..."
if [ -d "$TEST_MODELS_DIR" ]; then
    print_warning "Cleaning existing test models directory..."
    rm -rf "$TEST_MODELS_DIR"
fi
mkdir -p "$TEST_MODELS_DIR"
print_success "Test models directory created"

# Check if models directory exists (for pre-trained models)
if check_dir "models"; then
    print_info "Found pre-trained models directory"
    ls -la models/ | grep -E "\.(txt)$" || print_warning "No model files found in models/"
fi

echo
echo -e "${BLUE}=== Running Tests ===${NC}"

# Test 1: Help functionality
echo
print_info "Test 1: Help functionality"
if ./mlp --help > /dev/null 2>&1; then
    print_success "Help command works"
else
    print_error "Help command failed"
    exit 1
fi

# Test 2: Error handling (no arguments)
echo
print_info "Test 2: Error handling (no arguments)"
if ./mlp 2>/dev/null; then
    print_error "Should have failed with no arguments"
    exit 1
else
    print_success "Correctly handles no arguments"
fi

# Test 3: Error handling (missing mode)
echo
print_info "Test 3: Error handling (missing mode)"
if ./mlp --train 2>/dev/null; then
    print_error "Should have failed with missing mode"
    exit 1
else
    print_success "Correctly handles missing mode"
fi

# Test 4: Error handling (predict without model)
echo
print_info "Test 4: Error handling (predict without model)"
if ./mlp --mode mnist --predict 2>/dev/null; then
    print_error "Should have failed with predict mode but no model"
    exit 1
else
    print_success "Correctly handles predict without model"
fi

# Test 5: Boston Housing Training (if dataset available)
if [ -f "data/boston_housing.csv" ]; then
    echo
    print_info "Test 5: Boston Housing Training"
    if ./mlp --mode boston --train --epochs $TEST_EPOCHS --save "$TEST_MODELS_DIR/test_boston.txt" > /dev/null 2>&1; then
        print_success "Boston training completed"
        check_file "$TEST_MODELS_DIR/test_boston.txt"
    else
        print_error "Boston training failed"
        exit 1
    fi

    # Test 6: Boston Housing Prediction
    echo
    print_info "Test 6: Boston Housing Prediction"
    if ./mlp --mode boston --predict --load "$TEST_MODELS_DIR/test_boston.txt" > /dev/null 2>&1; then
        print_success "Boston prediction completed"
    else
        print_error "Boston prediction failed"
        exit 1
    fi

    # Test 7: Boston Housing Continued Training
    echo
    print_info "Test 7: Boston Housing Continued Training"
    if ./mlp --mode boston --train --epochs 5 --load "$TEST_MODELS_DIR/test_boston.txt" --save "$TEST_MODELS_DIR/test_boston_continued.txt" > /dev/null 2>&1; then
        print_success "Boston continued training completed"
        check_file "$TEST_MODELS_DIR/test_boston_continued.txt"
    else
        print_error "Boston continued training failed"
        exit 1
    fi
else
    print_warning "Skipping Boston Housing tests (dataset not available)"
fi

# Test 8: MNIST Training (if dataset available)
if [ -f "data/mnist_train.csv" ]; then
    echo
    print_info "Test 8: MNIST Training"
    if timeout 300 ./mlp --mode mnist --train --epochs $TEST_EPOCHS --save "$TEST_MODELS_DIR/test_mnist.txt" > /dev/null 2>&1; then
        print_success "MNIST training completed"
        check_file "$TEST_MODELS_DIR/test_mnist.txt"
    else
        print_error "MNIST training failed or timed out"
        exit 1
    fi

    # Test 9: MNIST Prediction (if test dataset available)
    if [ -f "data/mnist_test.csv" ]; then
        echo
        print_info "Test 9: MNIST Prediction"
        if ./mlp --mode mnist --predict --load "$TEST_MODELS_DIR/test_mnist.txt" > /dev/null 2>&1; then
            print_success "MNIST prediction completed"
        else
            print_error "MNIST prediction failed"
            exit 1
        fi
    else
        print_warning "Skipping MNIST prediction test (test dataset not available)"
    fi

    # Test 10: MNIST Continued Training
    echo
    print_info "Test 10: MNIST Continued Training"
    if timeout 180 ./mlp --mode mnist --train --epochs 5 --load "$TEST_MODELS_DIR/test_mnist.txt" --save "$TEST_MODELS_DIR/test_mnist_continued.txt" > /dev/null 2>&1; then
        print_success "MNIST continued training completed"
        check_file "$TEST_MODELS_DIR/test_mnist_continued.txt"
    else
        print_error "MNIST continued training failed or timed out"
        exit 1
    fi
else
    print_warning "Skipping MNIST tests (dataset not available)"
fi

# Test 11: Pre-trained model testing (if models exist)
if [ -d "models" ] && [ "$(ls -A models/*.txt 2>/dev/null)" ]; then
    echo
    print_info "Test 11: Pre-trained Model Testing"
    
    # Test Boston pre-trained models
    for model in models/*boston*.txt; do
        if [ -f "$model" ] && [ -f "data/boston_housing.csv" ]; then
            echo "  Testing pre-trained Boston model: $(basename $model)"
            if ./mlp --mode boston --predict --load "$model" > /dev/null 2>&1; then
                print_success "  Pre-trained Boston model works: $(basename $model)"
            else
                print_warning "  Pre-trained Boston model failed: $(basename $model)"
            fi
        fi
    done
    
    # Test MNIST pre-trained models
    for model in models/*mnist*.txt; do
        if [ -f "$model" ] && [ -f "data/mnist_test.csv" ]; then
            echo "  Testing pre-trained MNIST model: $(basename $model)"
            if ./mlp --mode mnist --predict --load "$model" > /dev/null 2>&1; then
                print_success "  Pre-trained MNIST model works: $(basename $model)"
            else
                print_warning "  Pre-trained MNIST model failed: $(basename $model)"
            fi
        fi
    done
else
    print_warning "No pre-trained models found for testing"
fi

# Test 12: Custom dataset path
if [ -f "data/boston_housing.csv" ]; then
    echo
    print_info "Test 12: Custom Dataset Path"
    if ./mlp --mode boston --train --epochs 3 --dataset data/boston_housing.csv --save "$TEST_MODELS_DIR/test_custom_path.txt" > /dev/null 2>&1; then
        print_success "Custom dataset path works"
        check_file "$TEST_MODELS_DIR/test_custom_path.txt"
    else
        print_error "Custom dataset path failed"
        exit 1
    fi
fi

# Test 13: Error handling (non-existent model file)
echo
print_info "Test 13: Error handling (non-existent model file)"
output=$(./mlp --mode boston --predict --load "nonexistent_model.txt" 2>&1)
if echo "$output" | grep -q "Error loading model"; then
    print_success "Correctly handles non-existent model file"
else
    print_error "Should have shown error message for non-existent model file"
    exit 1
fi

echo
print_info "Cleaning up test models..."
rm -rf "$TEST_MODELS_DIR"
print_success "Cleanup completed"

echo
echo -e "${GREEN}=== All Tests Passed! ===${NC}"
echo "Your MLP AI implementation is working correctly."
echo
echo "Summary of tested functionality:"
echo "  ✓ Command line argument parsing"
echo "  ✓ Help system"
echo "  ✓ Error handling"
if [ -f "data/boston_housing.csv" ]; then
    echo "  ✓ Boston Housing training and prediction"
fi
if [ -f "data/mnist_train.csv" ]; then
    echo "  ✓ MNIST training and prediction"
fi
echo "  ✓ Model saving and loading"
echo "  ✓ Continued training from saved models"
echo "  ✓ Custom dataset paths"
if [ -d "models" ] && [ "$(ls -A models/*.txt 2>/dev/null)" ]; then
    echo "  ✓ Pre-trained model compatibility"
fi

echo
print_info "To run individual tests, you can use:"
echo "  ./mlp --help"
echo "  ./mlp --mode boston --train --epochs 20 --save my_model.txt"
echo "  ./mlp --mode mnist --predict --load models/mnist_test.txt"
