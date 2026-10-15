# HDB Resale Classification

## Project Description
This project implements a machine learning pipeline to classify HDB (Housing & Development Board) resale flats as either above or below the median price in Singapore.
The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple regression models are trained and compared to find the best performing one based on various metrics.

## Prerequisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:

git clone https://github.com/jjwongjun/resale-classification.git
cd classification- hdb-resale

2. Create and activate a conda environment:
    ```
    conda create -n hdbenv python=3.11
    conda activate hdbenv
    ```

3. Install the required packages:

pip install -r requirements.txt

## Pipeline Execution
To run the complete pipeline, execute:

python main.py

This will:
1. Load and clean the data
2. Preprocess features
3. Split data into training, validation, and test sets
4. Train baseline models
5. Perform hyperparameter tuning
6. Evaluate models and select the best one

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Train/validation/test split ratios
- Hyperparameter grid for model tuning

### 2. Data Preparation (`DataPreparation` class)
The DataPreparation' class handles:
- Removing duplicates
- Standardizing flat type names
- Converting storey ranges to numerical values
- Filling missing town and flat model names
- Extracting year and month from date
- Converting remaining lease information to months
- Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)
The ModelTraining' class manages:
- Splitting data into training, validation, and test
sets
- Training baseline models (Logistic Regression, K-Nearest Neighbours (KNN), Decision Trees)
- Hyperparameter tuning for Logistic Regression, KNN, and Decision Trees models
- Evaluating models using multiple metrics (Accuracy, Precision, Recall, F1 Score, Receiver Operating Characteristic (ROC) Curve, Area Under Curve (AUC))
- Selecting the best model based on ROC-AUC score

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading configuration and data
- Initializing data preparation and model training
- Running the training and evaluation process
- Identifying and evaluating the best model on the test set

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an
exploratory data analysis (EDA) was conducted to
understand the data better and identify any potential
issues. Here are some key findings from the EDA:

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
- `price_category` encoded for binary classification
- **Numerical Features**: `floor_area_sqm`,
`remaining_lease_months`, `lease_commence_date`, `year`
- Standardized using `StandardScaler` because the numerical features have approximately normal distributions. In addition, the numerical features are on different scales, which can adversely impact the performance of machine learning algorithms if not addressed. Lastly, machine learning algorithms assume normally distributed data and are sensitive to feature scales, particularly those that rely on distance metrics
- **Nominal Features**: `month`, `town name`, `flatm_name` - Encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: `flat_type` - Encoded using `OrdinalEncoder` with predefined categories because flat types can be considered to have a natural ranking based on their size and number of rooms
- **Passthrough Features**: `storey_range` - Used as-is after converting to numerical values because already been ordinally encoded

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Logistic Regression**: Starting point for binary classification problems, providing a straightforward approach to predicting probabilities of class membership
2. **K-Nearest Neighbours (KNN)**: Non-parametric algorithm that classifies observations based on the majority class of their nearest neighbours in the feature space; can capture complex decision boundaries
3. **Decision Trees**: Non-parametric algorithm, make predictions by following a series of decisions based on the features; highly interpretable

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
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance.

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions (both true positives and true negatives) among the total number of cases
- **Precision**: Ratio of correctly predicted positive observations to the total predicted positives
- **Recall**: Ratio of correctly predicted positive observations to all observations
- **F1 Score**: Harmonic mean of Precision and Recall
- **Receiver Operating Characteristic (ROC) Curve**: Graphical representation that illustrates the performance of a binary classifier across different threshold values
- **Area Under Curve (AUC)**: Area under ROC curve

## Deployment Considerations
- Code is structured into modular well-organized componenets making code easier to maintain, more scalable, and ready for real world applications