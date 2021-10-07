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
temp_dir = ps.environment.runtime_temp_folder("tifs")
cm_dir = ps.environment.runtime_temp_folder("cm")

# Load data from SyncroSim
input_data = my_scenario.datasheets(name="InputData")

# Convert TIF files to numpy arrays using rasterio
n_samples = len(input_data.y)
for count, img in enumerate(input_data.X):
    with rasterio.open(os.path.join(temp_dir, img), "r") as raster:
        values = raster.read()
        if count == 0:
            X_array = np.empty((n_samples, values.shape[1] * values.shape[2]))
        # Flatten values into 1D arrays
        X_array[count] = values.reshape(-1)
            
# Split into train and test data
test_prop = my_scenario.datasheets(name="TrainTestSplit")
test_prop = test_prop.TestProp.item()
X_train, X_test, y_train, y_test = train_test_split(X_array,
                                                    input_data.y,
                                                    test_size=test_prop,
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
    "n_estimators": np.arange(n_estimators.Minimum.item(),
                              n_estimators.Maximum.item() + 1,
                              n_estimators.Step.item()),
    "max_features": np.arange(n_features.Minimum.item(),
                              n_features.Maximum.item() + 1,
                              n_features.Step.item()),
    "max_depth": np.arange(max_depth.Minimum.item(),
                           max_depth.Maximum.item() + 1,
                           max_depth.Step.item())}

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
individual_report.support = individual_report.support.astype(int)
individual_report.columns = ["TargetValue", "Precision", "Recall", "F1score",
                             "Support"]
my_scenario.save_datasheet(name="ModelPerformanceIndividual", 
                           data=individual_report)


# Save the overall model performance to a SyncroSim Datasheet
overall_performance = my_scenario.datasheets(name="ModelPerformanceOverall")
overall_performance.loc[0] = np.nan
overall_report = model_report.iloc[-3:]
overall_performance.Accuracy = np.round(overall_report.loc["accuracy"][0], 3)
overall_performance.MacroPrecision = np.round(
    overall_report.loc["macro avg", "precision"], 3)
overall_performance.WeightedPrecision = np.round(
    overall_report.loc["weighted avg", "precision"], 3)
overall_performance.MacroRecall = np.round(
    overall_report.loc["macro avg", "recall"], 3)
overall_performance.WeightedRecall = np.round(
    overall_report.loc["weighted avg", "recall"], 3)
overall_performance.MacroF1 = np.round(
    overall_report.loc["macro avg", "f1-score"], 3)
overall_performance.WeightedF1 = np.round(
    overall_report.loc["weighted avg", "f1-score"], 3)

# Set timesteps and iteration so that confusion matrix can be plotted
overall_performance.Iteration = 1
overall_performance.Timestep = 1

# Generate a confusion matrix as a heatmap raster and save as an external file
cm = metrics.confusion_matrix(y_test, y_pred)
with rasterio.open(os.path.join(cm_dir, "confusion_matrix.tif"),
                   mode="w", driver="GTiff", width=cm.shape[0],
                   height=cm.shape[1], count=1, 
                   dtype=np.int32) as infile:
    infile.write(cm, indexes=1)

overall_performance.ConfusionMatrix = os.path.join(cm_dir,
                                                   "confusion_matrix.tif")

# Save overall model performance Datasheet
my_scenario.save_datasheet(name="ModelPerformanceOverall",
                           data=overall_performance)