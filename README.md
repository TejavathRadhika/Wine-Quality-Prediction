# Wine Quality Prediction Using Linear Regression Machine Learning

This repository contains the code and resources for predicting the quality of wines using linear regression machine learning algorithms. The project focuses on analyzing various features of wines and training a linear regression model to predict their quality based on those features.

## Dataset

The dataset used for this project is the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) from the UCI Machine Learning Repository. It consists of red and white wine samples, each with 11 physicochemical features such as acidity, pH, alcohol content, etc., along with a quality rating ranging from 0 to 10. The dataset is available in the `data` directory.

## Dependencies

The following dependencies are required to run the code in this repository:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Code Structure

- `data/`: Directory containing the dataset files.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- `src/`: Python scripts for preprocessing, model training, and evaluation.
- `models/`: Directory to save trained models.
- `utils/`: Utility functions and classes used in the project.
- `README.md`: This file, providing an overview of the repository.

## Usage

To train and evaluate the linear regression model, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/wine-quality-prediction.git
   cd wine-quality-prediction
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Explore the Jupyter notebooks in the `notebooks/` directory to understand the project workflow and data analysis.

4. Preprocess the data:

   ```bash
   python src/data_preprocessing.py
   ```

   This script performs data cleaning, feature engineering, and splitting the dataset into training and testing sets.

5. Train the linear regression model:

   ```bash
   python src/train_model.py
   ```

   This script trains a linear regression model on the preprocessed data and saves it in the `models/` directory.

6. Evaluate the model:

   ```bash
   python src/evaluate_model.py
   ```

   This script evaluates the trained model using various metrics and generates visualizations.

## Results

The results of the linear regression model, including evaluation metrics and visualizations, can be found in the `results/` directory.

## Contributing

Contributions to this repository are always welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for personal and commercial purposes.
