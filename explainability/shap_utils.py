import shap
import pandas as pd
import numpy as np


def get_shap_values(model, X):
    """
    Compute SHAP values.

    Parameters
    ----------
    model : trained model
    X : pandas DataFrame

    Returns
    -------
    shap_values
    """

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)

    except Exception:

        explainer = shap.Explainer(model)

        shap_values = explainer(X)

    return shap_values


def get_top_features(model, X):
    """
    Return top features sorted by importance.
    """

    shap_values = get_shap_values(model, X)

    # New SHAP versions
    if hasattr(shap_values, "values"):

        values = shap_values.values

    else:

        values = shap_values

    # Handle binary classification
    if len(values.shape) == 3:

        values = values[:, :, 1]

    mean_abs = np.abs(values).mean(axis=0)

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": mean_abs

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    return importance


def get_customer_explanation(
    model,
    X,
    customer_index
):
    """
    Explain one customer.
    """

    shap_values = get_shap_values(
        model,
        X
    )

    if hasattr(shap_values, "values"):

        values = shap_values.values

    else:

        values = shap_values

    if len(values.shape) == 3:

        values = values[:, :, 1]

    customer_shap = values[customer_index]

    explanation = pd.DataFrame({

        "Feature": X.columns,

        "Impact": customer_shap

    })

    explanation["Absolute"] = (

        explanation["Impact"]

        .abs()

    )

    explanation = explanation.sort_values(

        by="Absolute",

        ascending=False

    )

    return explanation.drop(

        columns=["Absolute"]

    )