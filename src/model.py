# =============================================================================
# Run through a machine learning workflow
# =============================================================================

# Import basic packages
import pysyncrosim as ps
import rasterio
import numpy as np
import pandas as pd
import os

# Import machine learning packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

# =============================================================================
# Step 1: Load data, preprocess, split into train/test
# =============================================================================

# Retrieve the Scenario that is currently running
my_scenario = ps.Scenario()
env = ps.environment._environment()

# Load data from SyncroSim
input_data = my_scenario.datasheets(name="InputData")

# Create column for storing X values as arrays
input_data["X_array"] = np.nan

# Convert TIF files to numpy arrays using rasterio
for count, img in enumerate(input_data.X):
    with rasterio.open(img, "r") as raster:
        values = raster.read()
        if values.shape[0] == 1:
            input_data.loc[count, "X_array"] = values[0]
        else:
            input_data.loc[count, "X_array"] = values
            
# Split into train and test data
test_prop = my_scenario.datasheets(name="TrainTestSplit")
X_train, X_test, y_train, y_test = train_test_split(input_data.X_array,
                                                    input_data.y,
                                                    test_size=test_prop.item(),
                                                    shuffle=True)

# =============================================================================
# Step 2: Model and feature selection
# =============================================================================

# Initialize the random forest classifier model
random_forest = RandomForestClassifier()

# Load hyperparameters to optimize from SyncroSim
n_estimators = my_scenario.datasheets(name="NEstimators")
n_features = my_scenario.datasheets(name="NFeatures")
max_depth = my_scenario.datasheets(name="MaxDepth")

param_grid = {
    "n_estimators": np.arange(n_estimators.Minimum, n_estimators.Maximum + 1,
                              n_estimators.step),
    "max_features": np.arange(n_features.Minimum, n_features.Maximum + 1,
                              n_features.step),
    "max_depth": np.arange(max_depth.Minimum, max_depth.Maximum + 1,
                           max_depth.step)}

# Perform hyperparameter search
tune_iters = my_scenario.datasheets(name="TuningIterations")

random_search = RandomizedSearchCV(random_forest, param_grid,
                                   n_iter=tune_iters.nIterations.item())
random_search.fit(X_train, y_train)

# Add another datasheet for randomized search output??

# Set model to the best model found in randomized search
model = random_search.best_estimator_

# =============================================================================
# Step 3: Evaluate model
# =============================================================================

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict test targets based on test data
y_pred = model.predict(X_test)

# Save predicted and true y values to the Outputs Datasheet
outputs = pd.DataFrame({"yPred": y_pred, "yTest": y_test})
my_scenario.save_datasheet(name="Outputs", data=outputs)

# Generate a model performance report and put in a pandas DataFrame
model_report = pd.DataFrame(
    metrics.classification_report(y_test, y_pred, output_dict=True)).T

# Save the individual model performance to a SyncroSim Datasheet
individual_report = model_report.iloc[:-3].reset_index()
individual_report.columns = ["TargetValue", "Precision", "Recall", "F1score",
                             "Support"]
my_scenario.save_datasheet(name="ModelPerformanceIndividual", 
                           data=individual_report)


# Save the overall model performance to a SyncroSim Datasheet
overall_performance = my_scenario.datasheets(name="ModelPerformanceOverall")
overall_report = model_report.iloc[-3:]
overall_performance.Accuracy = np.round(overall_report.loc["accuracy"][0], 3)
overall_performance.MacroPrecision = np.round(
    overall_report.loc["macro avg", "precision"])
overall_performance.WeightedPrecision = np.round(
    overall_report.loc["weighted avg", "precision"])
overall_performance.MacroRecall = np.round(
    overall_report.loc["macro avg", "recall"])
overall_performance.WeightedRecall = np.round(
    overall_report.loc["weighted avg", "recall"])
overall_performance.MacroF1 = np.round(
    overall_report.loc["macro avg", "f1-score"])
overall_performance.WeightedF1= np.round(
    overall_report.loc["weighted avg", "f1-score"])

# Generate a confusion matrix as a heatmap raster and save as an external file
cm = metrics.confusion_matrix(y_test, y_pred)
with rasterio.open(os.path.join(env.output_directory, "confusion_matrix.tif"),
                   mode="w", driver="GTiff", width=cm.shape[0],
                   height=cm.shape[1], count=1, 
                   dtype=cm[0][0].dtype) as infile:
    infile.write(cm, indexes=1)
overall_performance.ConfusionMatrix = os.path.join(env.output_directory,
                                                   "confusion_matrix.tif")

# Save overall model performance Datasheet
my_scenario.save_datasheet(name="ModelPerformanceOverall",
                           data=overall_performance)