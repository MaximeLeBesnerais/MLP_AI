#!/bin/bash

# Test script to demonstrate behavior with missing datasets
# This temporarily moves datasets to show the warning messages

echo "=== Testing with Missing Datasets ==="

# Backup existing datasets
mkdir -p .dataset_backup
if [ -f "data/boston_housing.csv" ]; then
    mv data/boston_housing.csv .dataset_backup/
    echo "Temporarily moved Boston dataset"
fi

if [ -f "data/mnist_train.csv" ]; then
    mv data/mnist_train.csv .dataset_backup/
    echo "Temporarily moved MNIST training dataset"
fi

echo
echo "Running test with missing datasets..."
./test_mlp.sh || echo "Test completed with warnings as expected"

echo
echo "Restoring datasets..."
if [ -f ".dataset_backup/boston_housing.csv" ]; then
    mv .dataset_backup/boston_housing.csv data/
    echo "Restored Boston dataset"
fi

if [ -f ".dataset_backup/mnist_train.csv" ]; then
    mv .dataset_backup/mnist_train.csv data/
    echo "Restored MNIST training dataset"
fi

rmdir .dataset_backup 2>/dev/null || true

echo "Dataset test completed!"
