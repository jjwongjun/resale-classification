# HDB Resale Classification

## Project Description
This project implements a machine learning pipeline to classify HDB (Housing & Development Board) resale flats as either above or below the median price in Singapore.
The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple classification models are trained and compared to find the best performing one based on various metrics.

## Prerequisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
    ```
    git clone https://github.com/jjwongjun/resale-classification.git
    cd resale-classification
    ```

2. Create and activate a conda environment:
    ```
    conda create -n hdbenv python=3.11
    conda activate hdbenv
    ```

3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Pipeline Execution
To run the complete pipeline, execute:
```
python main.py
```
A different configuration file can be used with `python main.py --config path/to/config.yaml` (default: `./src/config.yaml`). The pipeline expects the data at `./data/resale_transactions_categorised.csv` (see `file_path` in the config).

This will:
1. Validate the configuration and load the data
2. Validate and clean the data
3. Split data into stratified training, validation, and test sets
4. Train baseline models (a dummy benchmark, linear, distance-based and tree-based models)
5. Perform hyperparameter tuning and report cross-validation results
6. Compare all models on the validation set and select the best one
7. Retrain the best model on training + validation data and evaluate it on the test set
8. Report feature importance of the final model

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml` and checks that all required keys exist and have valid types and values. The config includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Train/validation/test split ratios and the random seed
- Hyperparameter grids for model tuning
- Cross-validation folds and the scoring metric (`roc_auc`, the same metric used to select the best model)

### 2. Data Preparation (`DataPreparation` class)
The `DataPreparation` class handles:
- Logging a data validation report (duplicates, missing values per column, number of unique values in categorical columns)
- Removing duplicates
- Standardizing flat type names
- Converting storey ranges to numerical values
- Filling missing town and flat model names
- Extracting year and month from date
- Converting remaining lease information to months
- Encoding the target (`Above Median` = 1, `Below Median` = 0) and raising an error if an unexpected label is found
- Creating a preprocessor for feature transformation (including imputation of missing values)

### 3. Model Training (`ModelTraining` class)
The `ModelTraining` class manages:
- Splitting data into stratified training, validation, and test sets, and logging the class distribution of each
- Training baseline models (Dummy benchmark, Logistic Regression, K-Nearest Neighbours (KNN), Decision Trees, Random Forest, Histogram Gradient Boosting)
- Hyperparameter tuning for Logistic Regression, KNN, Decision Trees and Histogram Gradient Boosting models
- Reporting the mean and standard deviation of Accuracy, F1 and ROC AUC across the cross-validation folds for every model
- Evaluating models using multiple metrics (Accuracy, Precision, Recall, F1 Score, ROC AUC, PR AUC), where ROC AUC and PR AUC are computed from predicted probabilities
- Selecting the best model based on validation ROC-AUC score
- Retraining the selected model on training + validation data and reporting the confusion matrix and classification report on the test set
- Reporting permutation feature importance of the final model

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading and validating the configuration, and loading the data (with error handling)
- Initializing data preparation and model training
- Running the training and evaluation process
- Logging a side-by-side comparison of all models
- Identifying the best model, retraining it on training + validation data and evaluating it on the test set

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an exploratory data analysis (EDA) was conducted to understand the data better and identify any potential issues. Here are some key findings from the EDA:

**EDA Findings and Explanations**: 
- `town_name` and `flatm_name` features have null entries/missing values
- Duplicate rows 
- Balanced distribution of the two categories in `price_category`
- `flat_type` has labelling error of'4 ROOM' and 'FOUR ROOM' categories
- July 2018 shows the highest number of listings while February 2017 and January 2018 show the lowest number of listings compared to other months
- Number of listings drops significantly around February each year, because Feb is shorter month and BTO sales typically occurs in Feb
- Flat type increases in size, also spread of prices becomes wider
- IQR is wider for larger flat types like "EXECUTIVE" and "MULTI-GENERATION"
- Outlier 3-room flats are terrace houses
- Frequencies of 'Yishun', 'Bedok', and 'Punggol' do not match the frequencies of their respective numerical identifiers, which are 26, 2, and 18, respectively
- Towns with the highest number of resale units are Jurong West, Woodlands, and Sengkang; towns with the fewest resale units are Central Area, Marine Parade, and Bukit Timah
- Larger flat types (such as Executive and Multi-Generation flats) tend to cluster towards higher floor areas. Smaller flat types (like 1-room and 2-room flats) cluster towards lower floor areas

### Data Cleaning
- Storey ranges are converted to their average values because feature is stored as an object (string) and conversion maintains ordinal nature
- Remaining lease information is extracted and converted to total months because feature is stored as an object (string) with various inconsistent formats
- Year and month are extracted from the transaction date because feature is stored as an object (string) that contains both year and month format
- Missing town and flat model names are filled using ID-to-name mappings because missing values need to be accurately and systematically addressed, for machine learning algorithm use

### Feature Processing
- `price_category` encoded for binary classification. The label is already provided in the dataset (relative to the median price), so no threshold is computed in the pipeline
- Missing values are imputed (median for numerical features, most frequent category for nominal and ordinal features) inside the preprocessor, so the imputation values are learned from the training data only
- The classes are balanced (about 50% / 50%), so no resampling or class weights are used. The class distribution of each split is logged to confirm that the stratified split preserves it
- **Numerical Features**: `floor_area_sqm`,
`remaining_lease_months`, `lease_commence_date`, `year`
- Standardized using `StandardScaler` because the numerical features are on different scales, which can adversely impact the performance of scale-sensitive algorithms (Logistic Regression and KNN) if not addressed. Tree-based models are not affected by scaling
- **Nominal Features**: `month`, `town_name`, `flatm_name` - Encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: `flat_type` - Encoded using `OrdinalEncoder` with predefined categories because flat types can be considered to have a natural ranking based on their size and number of rooms
- **Passthrough Features**: `storey_range` - Used as-is after converting to numerical values because already been ordinally encoded

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Logistic Regression**: Starting point for binary classification problems, providing a straightforward approach to predicting probabilities of class membership
2. **K-Nearest Neighbours (KNN)**: Non-parametric algorithm that classifies observations based on the majority class of their nearest neighbours in the feature space; can capture complex decision boundaries
3. **Decision Trees**: Non-parametric algorithm, make predictions by following a series of decisions based on the features; highly interpretable
4. **Random Forest**: An ensemble of decision trees trained on random subsets of the data, which reduces the overfitting of a single tree
5. **Histogram Gradient Boosting**: Trees built one after another, each correcting the errors of the previous ones; usually among the most accurate methods for tabular data and fast on large datasets
6. **Dummy Classifier**: Always predicts the class prior. It is the minimum benchmark, so every real model must beat it (ROC AUC of 0.5) to be useful

### Hyperparameter Tuning
- Grid search is performed for Logistic Regression, KNN, and Decision Trees models because tuning model parameters can achieve the best possible performance. Grid Search CV's process is exhaustive but it ensures the best combination
- Parameters tuned because the right combination of these settings ensures balance and should perform well on unseen data. These include:
Logistic Regression:
- C (inverse of regularization strength)
- penalty (norm used in the penalization, e.g., 'l2')
K-Nearest Neighbors (KNN):
- n_neighbors (number of neighbors to use)
- weights (uniform or distance)
Decision Tree:
- max_depth (maximum depth of the tree)
- min_samples_split (minimum number of samples required to split an internal node)
- min_samples_leaf (minimum number of samples required to be at a leaf node)
Histogram Gradient Boosting:
- learning_rate (contribution of each tree)
- max_iter (number of boosting iterations)
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance. The same cross-validation summary is also reported for the baseline models, so they are not judged on a single validation split only
- Grid search is scored with ROC AUC (`scoring: roc_auc`), the same metric used to select the best model
- The best model is selected on the validation set and then retrained on training + validation data before it is evaluated once on the test set

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions (both true positives and true negatives) among the total number of cases
- **Precision**: Ratio of correctly predicted positive observations to the total predicted positives
- **Recall**: Ratio of correctly predicted positive observations to all actual positive observations
- **F1 Score**: Harmonic mean of Precision and Recall
- **Receiver Operating Characteristic (ROC) Curve**: Graphical representation that illustrates the performance of a binary classifier across different threshold values
- **Area Under Curve (AUC)**: Area under ROC curve, computed from predicted probabilities
- **PR AUC**: Area under the precision-recall curve (average precision), also computed from predicted probabilities
- **Confusion matrix and classification report**: Reported for the best model on the test set

### Results
Validation set (the final test set is only used once, for the best model):

| Model | Accuracy | F1 Score | ROC AUC |
|---|---|---|---|
| Histogram Gradient Boosting (tuned) | 0.923 | 0.923 | 0.980 |
| Histogram Gradient Boosting | 0.917 | 0.917 | 0.978 |
| Random Forest | 0.920 | 0.920 | 0.977 |
| Logistic Regression (tuned) | 0.893 | 0.892 | 0.962 |
| KNN (tuned) | 0.879 | 0.877 | 0.952 |
| Decision Tree (tuned) | 0.878 | 0.871 | 0.950 |
| Dummy benchmark | 0.503 | 0.000 | 0.500 |

All real models clearly beat the dummy benchmark, and the tree-based ensembles perform best. The best model, the tuned Histogram Gradient Boosting model, scores **Accuracy 0.935, Precision 0.936, Recall 0.933, F1 0.934 and ROC AUC 0.985** on the test set after being retrained on training + validation data. The confusion matrix shows similar numbers of false positives (267) and false negatives (283), so the model is not biased towards one class. Permutation importance shows that `floor_area_sqm` and `town_name` are by far the most important features, followed by `lease_commence_date`, `flat_type` and `storey_range`.

## Deployment Considerations
- Code is structured into modular well-organized components making code easier to maintain, more scalable, and ready for real world applications
- The configuration is validated and file loading, cleaning and training errors are logged with clear messages
- Possible next steps: unit tests for the cleaning functions, saving the fitted model with its metadata and a prediction mode for new data