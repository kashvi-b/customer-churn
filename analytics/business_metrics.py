def calculate_business_impact(df):

    high_risk = df[

        df["risk_tier"]

        == "High"

    ]

    total_customers = len(df)

    high_risk_customers = len(

        high_risk

    )

    potential_loss = abs(
        high_risk["MonthlyCharges"]
    ).sum()

    return {

        "total_customers":

        total_customers,

        "high_risk_customers":

        high_risk_customers,

        "potential_loss":

        potential_loss

    }