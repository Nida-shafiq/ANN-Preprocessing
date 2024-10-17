# ANN-Preprocessing

Creating a Custom Dataset with PyTorch:

 A Step-by-Step Guide

In this article, we'll walk through the process of creating a custom dataset class in PyTorch. This guide will demonstrate how to load a dataset from a CSV file, preprocess it, and prepare it for use in machine learning tasks. We will cover key concepts such as label encoding for categorical variables, handling missing data, and creating a data loader for efficient batch processing.

Prerequisites

Before you begin, ensure you have the following installed:

- Python (3.8 or later recommended)
- PyTorch
- Pandas

You can install PyTorch and Pandas using pip:

```bash
pip install torch torchvision torchaudio pandas
```

 Dataset Overview

We will work with a dataset in CSV format. The dataset contains various features, including categorical and numerical data, along with a target variable. Our goal is to preprocess this dataset to prepare it for training machine learning models.

## Step 1: Import Libraries

First, we need to import the necessary libraries:

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
```

## Step 2: Create the Custom Dataset Class

Next, we'll create a `CustomDataset` class that inherits from `torch.utils.data.Dataset`. This class will handle loading the data, preprocessing it, and providing it in a format suitable for training.

```python
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        # Load the dataset
        self.data = pd.read_csv(csv_file, delimiter=';')
        
        # Convert categorical columns to numeric (label encoding)
        categorical_columns = [
            'Marital status',
            'Application mode',
            'Course',
            'Daytime/evening attendance',
            'Previous qualification',
            'Nacionality',
            "Mother's qualification",
            "Father's qualification",
            "Mother's occupation",
            "Father's occupation",
            'Displaced',
            'Educational special needs',
            'Debtor',
            'Tuition fees up to date',
            'Gender',
            'Scholarship holder',
            'International'
        ]

        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype('category').cat.codes
        
        # Define features and target
        self.X = self.data.drop('Target', axis=1)  # Features
        self.y = self.data['Target']                # Target

        # Convert to numeric, handling NaNs
        self.X = self.X.apply(pd.to_numeric, errors='coerce')
        self.y = pd.to_numeric(self.y, errors='coerce')

        # Drop NaNs
        self.data = self.data.dropna(subset=['Target'])
        self.X = self.X.loc[self.data.index]
        self.y = self.y.loc[self.data.index]
```

### Explanation of the `CustomDataset` Class

1. **Loading the Dataset**: The dataset is loaded from a CSV file using `pd.read_csv()`. The `delimiter` is specified as `;`, as required by the dataset format.
  
2. **Label Encoding**: We convert categorical variables to numeric codes using `astype('category').cat.codes`. This step is crucial as machine learning models require numerical input.

3. **Defining Features and Target**: The features (independent variables) are stored in `self.X`, and the target (dependent variable) is stored in `self.y`.

4. **Handling Missing Data**: We convert all columns to numeric, coercing errors to NaN. Then, we drop rows with NaN values from both `self.X` and `self.y` to maintain alignment.

## Step 3: Implement the Data Loader

The `__len__` and `__getitem__` methods are implemented to provide a way to access the dataset in a format that PyTorch can use.

```python
    def __len__(self):
        return len(self.y)  # Return the length of the target array

    def __getitem__(self, idx):
        x_sample = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)  # Feature
        y_sample = torch.tensor(self.y.iloc[idx], dtype=torch.float32)  # Target
        return x_sample, y_sample
```

### Explanation of Methods

- `__len__`: Returns the number of samples in the dataset.
- `__getitem__`: Given an index, it retrieves the corresponding feature and target data, converting them to PyTorch tensors.

## Step 4: Using the Custom Dataset

Finally, we can utilize the `CustomDataset` class and create a DataLoader to facilitate batch processing.

```python
if __name__ == "__main__":
    # Create an instance of the dataset
    dataset = CustomDataset('data.csv')
    
    # Print the length of the dataset
    print(f"Dataset size: {len(dataset)}")
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through the DataLoader and print a batch
    for x_batch, y_batch in dataloader:
        print("X batch:", x_batch)
        print("Y batch:", y_batch)
        break  # Remove this line if you want to see more batches
```

### Explanation of Main Block

1. **Creating the Dataset Instance**: We instantiate `CustomDataset` with the path to the CSV file.
  
2. **Creating the DataLoader**: The DataLoader handles batching and shuffling of data for training. 

3. **Iterating Through Batches**: We iterate through the DataLoader to fetch batches of data.

## Conclusion

In this article, we demonstrated how to create a custom dataset in PyTorch using a CSV file. We covered the essential preprocessing steps, including label encoding and handling missing values, and provided a framework for loading the data efficiently. This setup allows you to integrate your dataset into machine learning models easily.

Feel free to modify the code and adapt it to your specific needs. Happy coding!

--- 

You can adjust the content as needed, but this should provide a solid foundation for your GitHub repository.
