# Explainability in ML Models

In this [notebook](explainability.ipynb), we give some intuition behind explainability in black-box models. We start from the simple case of linear regression, and then progress to a more complex model based on LightGBM, all grounded on the California Housing dataset as ground truth.

## Dataset

The [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) describes house prices across California block groups. Each sample is characterised by the following variables:

| Variable | Type | Description |
|---|---|---|
| MedInc | float | Median income in block |
| HouseAge | float | Median house age in block |
| AveRooms | float | Average rooms in dwelling |
| AveBedrms | float | Average bedrooms in dwelling |
| Population | float | Block population |
| AveOccup | float | Average house occupancy |
| Latitude | float | House block latitude |
| Longitude | float | House block longitude |

## Dependencies

The notebook was created with Python 3.12.12. Install the pinned dependencies with:

```bash
pip install -r requirements.txt
```

## Overview

### Linear Regression

Linear regression is a simple yet powerful model that is particularly easy to interpret. After the model is fit, each variable is associated with a parameter (coefficient) that it is multiplied by, transforming its value into the units of the variable of interest. The sum of all these transformed variables, plus an offset (intercept), is what produces the final prediction.

This initial interpretation is useful; however, there is an important limitation: the scale of the parameters can be misleading without context. If variables are not normalised, those with large numerical scales will tend to have smaller coefficients, and vice versa. A variable like `HouseAge` measured in seconds instead of years would have a much smaller coefficient — not because it matters less, but simply because its input values are much larger.

### SHAP Values

This is where SHAP comes in. SHAP (Shapley values) is a method based on game theory that allows us to understand the marginal contribution of each feature to a model's prediction. Unlike linear regression coefficients, which provide a single global interpretation, SHAP allows us to compute per-sample explanations, showing how each feature contributes to an individual prediction, not just on average across the dataset.

We cover four SHAP visualisations:

- **Dependence plot** — shows how the model output changes as a feature varies, as well as how frequently different values of that feature occur in the data. This helps us see not only the effect of a feature, but also how relevant that effect is in practice.

- **Scatter plot** — places feature values on the x-axis and their corresponding SHAP values on the y-axis, showing how changes in the feature affect the prediction. By passing `shap_values` to the colour argument, SHAP automatically selects the feature that is most strongly correlated (or interacting) with the SHAP values of the selected feature, and uses it to colour the points.

- **Waterfall plot** — gives a per-sample explanation of the model's prediction, showing how each variable contributed to the final output for a single observation, rather than how a variable behaves across the entire dataset. Each bar shows how a feature moves the prediction away from the baseline expected value to reach the final output.

- **Beeswarm plot** — shows the SHAP values of every variable across all samples. Each point represents a sample, positioned according to its SHAP value (impact on the model output), while the colour indicates the value of the feature (red = high value, blue = low value). This allows us to understand both how the value of a feature influences the prediction, and how common different effects are.

## Usage

Open and run `explainability.ipynb` in Jupyter or on Kaggle.

## Use of AI

I recognize the usage of copilot to help me write this Readme and the requirements.txt, which were manually tested to re run the jupyter notebook.
---