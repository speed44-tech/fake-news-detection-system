def calculate_overall_risk(
    svm_label,
    svm_confidence,
    transformer_label,
    transformer_confidence,
    suspicious_result,
    claim_result,
    source_result,
    evidence_result,
    model_agreement
):
    """
    Combines multiple analysis signals into an overall
    news-risk assessment.

    This score is an AI risk indicator.
    It is NOT a factual truth probability.
    """

    risk_score = 0
    reasons = []

    # ======================================================
    # AI MODEL SIGNAL
    # ======================================================

    if svm_label == 0 and transformer_label == 0:

        risk_score += 40

        reasons.append(
            "Both AI models classified the content as likely unreliable."
        )

    elif svm_label == 0 or transformer_label == 0:

        risk_score += 20

        reasons.append(
            "One AI model classified the content as likely unreliable."
        )

    else:

        reasons.append(
            "Both AI models classified the content as likely reliable."
        )


    # ======================================================
    # MODEL CONFIDENCE
    # ======================================================

    average_model_confidence = (
        svm_confidence + transformer_confidence
    ) / 2


    if average_model_confidence < 60:

        risk_score += 15

        reasons.append(
            "The models show relatively low confidence."
        )

    elif average_model_confidence < 75:

        risk_score += 8

        reasons.append(
            "The models show moderate confidence."
        )


    # ======================================================
    # MODEL AGREEMENT
    # ======================================================

    if model_agreement == "Strong Model Agreement":

        reasons.append(
            "The machine-learning and transformer models agree."
        )

    else:

        risk_score += 15

        reasons.append(
            "The machine-learning and transformer models disagree."
        )


    # ======================================================
    # SUSPICIOUS CONTENT
    # ======================================================

    suspicious_score = suspicious_result.get(
        "risk_score",
        0
    )


    risk_score += round(
        suspicious_score * 0.20
    )


    if suspicious_score >= 50:

        reasons.append(
            "Multiple suspicious linguistic signals were detected."
        )

    elif suspicious_score >= 25:

        reasons.append(
            "Some suspicious linguistic signals were detected."
        )


    # ======================================================
    # CLAIM ANALYSIS
    # ======================================================

    verification_score = claim_result.get(
        "verification_score",
        0
    )


    risk_score += round(
        verification_score * 0.15
    )


    if verification_score >= 60:

        reasons.append(
            "The content contains claims that may require "
            "additional verification."
        )

    elif verification_score >= 30:

        reasons.append(
            "Some claims may require additional verification."
        )


    # ======================================================
    # SOURCE ANALYSIS
    # ======================================================

    source_score = source_result.get(
        "source_score",
        0
    )


    risk_score += round(
        source_score * 0.15
    )


    if source_score >= 50:

        reasons.append(
            "The submitted source contains several structural "
            "URL risk indicators."
        )

    elif source_score >= 25:

        reasons.append(
            "The submitted source contains some structural "
            "URL risk indicators."
        )


    # ======================================================
    # EVIDENCE ANALYSIS
    # ======================================================

    evidence_available = evidence_result.get(
        "evidence_available",
        False
    )


    evidence_status = evidence_result.get(
        "evidence_status",
        "No evidence retrieved"
    )


    keyword_match_score = evidence_result.get(
        "match_score",
        0
    )


    semantic_score = evidence_result.get(
        "semantic_score",
        0
    )


    combined_score = evidence_result.get(
        "combined_score",
        0
    )


    # ======================================================
    # EVIDENCE SIGNAL
    # ======================================================

    if evidence_available:

        # --------------------------------------------------
        # EXPLICIT CONTRADICTION
        # --------------------------------------------------

        if evidence_status == (
            "Potential contradiction detected"
        ):

            risk_score += 35

            reasons.append(
                "Retrieved source evidence contains "
                "fact-checking or contradiction-related language."
            )


        # --------------------------------------------------
        # STRONG SEMANTIC EVIDENCE
        # --------------------------------------------------

        elif combined_score >= 75:

            risk_score += 3

            reasons.append(
                "Retrieved evidence is strongly related "
                "to the submitted claim."
            )


        # --------------------------------------------------
        # MODERATE SEMANTIC EVIDENCE
        # --------------------------------------------------

        elif combined_score >= 50:

            risk_score += 6

            reasons.append(
                "Retrieved evidence shows moderate "
                "relevance to the submitted claim."
            )


        # --------------------------------------------------
        # LIMITED EVIDENCE
        # --------------------------------------------------

        elif combined_score >= 25:

            risk_score += 10

            reasons.append(
                "Retrieved evidence has limited relevance "
                "to the submitted claim."
            )


        # --------------------------------------------------
        # WEAK EVIDENCE
        # --------------------------------------------------

        else:

            risk_score += 12

            reasons.append(
                "Retrieved evidence has weak relevance "
                "to the submitted claim."
            )


        # ==================================================
        # SEMANTIC EVIDENCE INFORMATION
        # ==================================================

        if semantic_score >= 75:

            reasons.append(
                "Semantic analysis found strong contextual "
                "similarity between the claim and source."
            )

        elif semantic_score >= 50:

            reasons.append(
                "Semantic analysis found moderate contextual "
                "similarity between the claim and source."
            )

        elif semantic_score > 0:

            reasons.append(
                "Semantic analysis found limited contextual "
                "similarity between the claim and source."
            )


        # ==================================================
        # KEYWORD EVIDENCE INFORMATION
        # ==================================================

        if keyword_match_score >= 50:

            reasons.append(
                "The submitted claim shares substantial "
                "terminology with the retrieved source."
            )


    else:

        reasons.append(
            "No readable evidence was retrieved from the "
            "submitted source."
        )


    # ======================================================
    # LIMIT SCORE
    # ======================================================

    risk_score = max(
        0,
        min(
            100,
            risk_score
        )
    )


    # ======================================================
    # FINAL INTELLIGENCE LEVEL
    # ======================================================

    if risk_score >= 70:

        intelligence_level = "High Risk"

        recommendation = (
            "Additional verification is strongly recommended "
            "before treating this information as reliable."
        )


    elif risk_score >= 45:

        intelligence_level = "Moderate Risk"

        recommendation = (
            "Review the supporting evidence and source before "
            "accepting the information as reliable."
        )


    elif risk_score >= 25:

        intelligence_level = "Low-Moderate Risk"

        recommendation = (
            "Some signals require attention, but the available "
            "analysis does not indicate a high level of risk."
        )


    else:

        intelligence_level = "Low Risk Indicators"

        recommendation = (
            "No major risk indicators were detected by the "
            "current analysis layers."
        )


    # ======================================================
    # RETURN RESULT
    # ======================================================

    return {

        "risk_score": risk_score,

        "intelligence_level": intelligence_level,

        "recommendation": recommendation,

        "average_model_confidence": round(
            average_model_confidence,
            1
        ),

        "reasons": reasons

    } 