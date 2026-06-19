def retention_recommendation(customer):

    actions = []

    if customer["tenure"] < 12:

        actions.append(
            "📞 Assign onboarding specialist"
        )

    if customer["MonthlyCharges"] > 80:

        actions.append(
            "💰 Offer customized pricing plan"
        )

    if customer["is_monthly"] == 1:

        actions.append(
            "📝 Offer yearly contract discount"
        )

    if customer["no_support"] == 1:

        actions.append(
            "🎧 Provide priority customer support"
        )

    if len(actions) == 0:

        actions.append(
            "✅ No immediate action required"
        )

    return ", ".join(actions)