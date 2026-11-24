from flask import Flask, render_template, request
import joblib
from pathlib import Path
import math
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from decision_engine import combine_predictions
from suspicious_signals import analyze_suspicious_signals
from claim_analysis import analyze_claims
from source_analysis import analyze_source
from evidence_analysis import analyze_evidence
from risk_engine import calculate_overall_risk


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOAD SVM MODEL + TF-IDF
# ============================================================

model = joblib.load(
    BASE_DIR / "model.pkl"
)

vectorizer = joblib.load(
    BASE_DIR / "vectorizer.pkl"
)

print("SVM model loaded successfully!")
print("TF-IDF vectorizer loaded successfully!")


# ============================================================
# LOAD TRANSFORMER
# ============================================================

TRANSFORMER_NAME = "GhostFaith/distilbert-fakenews"

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Loading Transformer...")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(
    TRANSFORMER_NAME
)

transformer = AutoModelForSequenceClassification.from_pretrained(
    TRANSFORMER_NAME
).to(device)

transformer.eval()

print("Transformer loaded successfully!")


# ============================================================
# SVM CONFIDENCE
# ============================================================

def calculate_svm_confidence(decision_score):

    confidence = 1 / (
        1 +
        math.exp(
            -abs(
                float(decision_score)
            )
        )
    )

    percentage = confidence * 100

    percentage = max(
        50.0,
        min(
            99.0,
            percentage
        )
    )

    return round(
        percentage,
        1
    )


# ============================================================
# TRANSFORMER PREDICTION
# ============================================================

def get_transformer_prediction(news_text):

    encoded = tokenizer(
        news_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # DistilBERT does not need token_type_ids
    encoded.pop(
        "token_type_ids",
        None
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = transformer(
            **encoded
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    raw_prediction = int(
        torch.argmax(
            probabilities
        ).item()
    )

    confidence = float(
        probabilities[
            raw_prediction
        ].item()
        * 100
    )

    # Convert DistilBERT labels to the project's convention:
    # 0 = Fake, 1 = Real
    # DistilBERT checkpoint: 0 = Real, 1 = Fake
    prediction = 1 - raw_prediction

    return (
        prediction,
        round(
            confidence,
            1
        )
    )


# ============================================================
# HOME ROUTE
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def home():

    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    prediction = None
    prediction_label = None

    confidence = None
    confidence_text = None

    news_text = ""
    source_url = ""

    error_message = None

    svm_label = None
    svm_confidence = None

    transformer_label = None
    transformer_confidence = None

    model_agreement = None

    suspicious_result = None
    claim_result = None
    source_result = None
    evidence_result = None
    risk_result = None


    # ========================================================
    # POST REQUEST
    # ========================================================

    if request.method == "POST":

        # ----------------------------------------------------
        # GET USER INPUT
        # ----------------------------------------------------

        news_text = request.form.get(
            "news",
            ""
        ).strip()

        source_url = request.form.get(
            "source_url",
            ""
        ).strip()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not news_text:

            error_message = (
                "Please enter a news headline or article "
                "before analyzing."
            )

        elif len(news_text) < 30:

            error_message = (
                "Please enter more complete news content. "
                "A headline or article should contain more "
                "information."
            )

        else:

            # =================================================
            # 1. SOURCE ANALYSIS
            # =================================================

            source_result = analyze_source(
                source_url
            )


            # =================================================
            # 2. SUSPICIOUS LANGUAGE ANALYSIS
            # =================================================

            suspicious_result = (
                analyze_suspicious_signals(
                    news_text
                )
            )


            # =================================================
            # 3. CLAIM ANALYSIS
            # =================================================

            claim_result = analyze_claims(
                news_text
            )


            # =================================================
            # 4. EVIDENCE RETRIEVAL
            # =================================================

            evidence_result = analyze_evidence(
                news_text,
                source_url
            )


            # =================================================
            # 5. SVM PREDICTION
            # =================================================

            news_tfidf = vectorizer.transform(
                [news_text]
            )

            svm_label = int(
                model.predict(
                    news_tfidf
                )[0]
            )

            svm_decision_score = (
                model.decision_function(
                    news_tfidf
                )[0]
            )

            svm_confidence = (
                calculate_svm_confidence(
                    svm_decision_score
                )
            )


            # =================================================
            # 6. TRANSFORMER PREDICTION
            # =================================================

            (
                transformer_label,
                transformer_confidence
            ) = get_transformer_prediction(
                news_text
            )


            # =================================================
            # 7. COMBINE AI MODELS
            # =================================================

            result = combine_predictions(
                svm_label,
                svm_confidence,
                transformer_label,
                transformer_confidence
            )

            prediction_label = result[
                "label"
            ]

            confidence = result[
                "confidence"
            ]

            model_agreement = result[
                "agreement"
            ]

            prediction = (
                "Model Assessment: "
                +
                result["assessment"]
            )

            confidence_text = (
                result["confidence_level"]
                +
                " model confidence"
            )


            # =================================================
            # 8. OVERALL RISK ENGINE
            # =================================================

            risk_result = calculate_overall_risk(
                svm_label,
                svm_confidence,
                transformer_label,
                transformer_confidence,
                suspicious_result,
                claim_result,
                source_result,
                evidence_result,
                model_agreement
            )


            # =================================================
            # TERMINAL ANALYSIS OUTPUT
            # =================================================

            print(
                "\n=============================="
            )

            print(
                "AI NEWS ANALYSIS"
            )

            print(
                "=============================="
            )


            # -------------------------------------------------
            # MODEL RESULTS
            # -------------------------------------------------

            print(
                "SVM:",
                svm_label,
                "|",
                svm_confidence,
                "%"
            )

            print(
                "Transformer:",
                transformer_label,
                "|",
                transformer_confidence,
                "%"
            )

            print(
                "Agreement:",
                model_agreement
            )

            print(
                "Final:",
                prediction
            )

            print(
                "Final confidence:",
                confidence,
                "%"
            )


            # -------------------------------------------------
            # SUSPICIOUS SIGNALS
            # -------------------------------------------------

            print(
                "Suspicious signal score:",
                suspicious_result[
                    "risk_score"
                ]
            )

            print(
                "Signal level:",
                suspicious_result[
                    "signal_level"
                ]
            )


            # -------------------------------------------------
            # CLAIM ANALYSIS
            # -------------------------------------------------

            print(
                "Claims detected:",
                claim_result[
                    "claim_count"
                ]
            )

            print(
                "Numerical claims:",
                claim_result[
                    "numerical_claims"
                ]
            )

            print(
                "Verification need:",
                claim_result[
                    "verification_status"
                ]
            )


            # -------------------------------------------------
            # SOURCE ANALYSIS
            # -------------------------------------------------

            print(
                "Source:",
                source_result[
                    "domain"
                ]
            )

            print(
                "Source risk:",
                source_result[
                    "source_score"
                ]
            )

            print(
                "Source level:",
                source_result[
                    "source_level"
                ]
            )


            # -------------------------------------------------
            # EVIDENCE ANALYSIS
            # -------------------------------------------------

            print(
                "Evidence available:",
                evidence_result[
                    "evidence_available"
                ]
            )

            print(
                "Evidence status:",
                evidence_result[
                    "evidence_status"
                ]
            )

            print(
                "Evidence match score:",
                evidence_result[
                    "match_score"
                ]
            )

            print(
                "Semantic evidence score:",
                evidence_result[
                    "semantic_score"
                ]
            )

            print(
                "Combined evidence score:",
                evidence_result[
                    "combined_score"
                ]
            )

            print(
                "Evidence strength:",
                evidence_result[
                    "evidence_strength"
                ]
            )

            print(
                "Source title:",
                evidence_result[
                    "source_title"
                ]
            )

            print(
                "Matched terms:",
                len(
                    evidence_result[
                        "matched_terms"
                    ]
                )
            )


            # -------------------------------------------------
            # OVERALL RISK
            # -------------------------------------------------

            print(
                "Overall risk:",
                risk_result[
                    "risk_score"
                ]
            )

            print(
                "Intelligence level:",
                risk_result[
                    "intelligence_level"
                ]
            )

            print(
                "Recommendation:",
                risk_result[
                    "recommendation"
                ]
            )

            print(
                "==============================\n"
            )


    # ========================================================
    # RENDER HTML
    # ========================================================

    return render_template(
        "index.html",

        prediction=prediction,
        prediction_label=prediction_label,

        confidence=confidence,
        confidence_text=confidence_text,

        news_text=news_text,
        source_url=source_url,

        error_message=error_message,

        svm_label=svm_label,
        svm_confidence=svm_confidence,

        transformer_label=transformer_label,
        transformer_confidence=transformer_confidence,

        model_agreement=model_agreement,

        suspicious_result=suspicious_result,

        claim_result=claim_result,

        source_result=source_result,

        evidence_result=evidence_result,

        risk_result=risk_result
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5050
    )