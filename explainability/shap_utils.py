import shap
import pandas as pd
import numpy as np


def get_shap_values(model, X):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    return shap_values


def get_top_features(model, X):

    shap_values = get_shap_values(
        model,
        X
    )

    # Handle binary classification
    if len(np.array(shap_values).shape) == 3:

        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(
        shap_values
    ).mean(axis=0)

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

    shap_values = get_shap_values(
        model,
        X
    )

    if len(np.array(shap_values).shape) == 3:

        shap_values = shap_values[:, :, 1]

    customer_shap = shap_values[customer_index]

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