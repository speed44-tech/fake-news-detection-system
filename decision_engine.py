def combine_predictions(
    svm_label,
    svm_confidence,
    transformer_label,
    transformer_confidence
):
    """
    Combines predictions from:
    1. Linear SVM
    2. DistilBERT Transformer

    Labels:
        0 = Fake / Unreliable
        1 = Real / Reliable
    """

    # --------------------------------------------------
    # CASE 1: Both models agree
    # --------------------------------------------------

    if svm_label == transformer_label:

        final_label = svm_label

        # Average the two model confidence indicators
        final_confidence = (
            svm_confidence + transformer_confidence
        ) / 2

        agreement = "Strong Model Agreement"

        if final_label == 1:
            assessment = "Likely Reliable"
        else:
            assessment = "Likely Unreliable"

    # --------------------------------------------------
    # CASE 2: Models disagree
    # --------------------------------------------------

    else:

        agreement = "Model Disagreement"

        # Use the stronger model confidence
        if transformer_confidence >= svm_confidence:
            final_label = transformer_label
            final_confidence = transformer_confidence
        else:
            final_label = svm_label
            final_confidence = svm_confidence

        # Reduce confidence because the models disagree
        final_confidence *= 0.75

        if final_label == 1:
            assessment = "Possibly Reliable"
        else:
            assessment = "Possibly Unreliable"

    # --------------------------------------------------
    # Confidence category
    # --------------------------------------------------

    if final_confidence >= 90:
        confidence_level = "High"
    elif final_confidence >= 75:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    return {
        "label": final_label,
        "assessment": assessment,
        "confidence": round(final_confidence, 1),
        "confidence_level": confidence_level,
        "agreement": agreement
    }