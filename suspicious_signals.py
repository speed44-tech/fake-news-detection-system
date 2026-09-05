import re


SENSATIONAL_PHRASES = [
    "shocking",
    "breaking",
    "you won't believe",
    "must see",
    "urgent",
    "secret revealed",
    "they don't want you to know",
    "100% true",
    "guaranteed",
    "miracle",
    "exposed",
    "wake up",
    "share this",
    "viral",
    "huge scandal",
    "unbelievable",
]


def analyze_suspicious_signals(text):
    signals = []
    score = 0

    text = text.strip()

    if not text:
        return {
            "risk_score": 0,
            "signals": [],
            "signal_level": "Low"
        }

    # --------------------------------
    # 1. Excessive exclamation marks
    # --------------------------------
    exclamation_count = text.count("!")

    if exclamation_count >= 3:
        signals.append(
            f"Excessive exclamation marks detected ({exclamation_count})"
        )
        score += 15

    # --------------------------------
    # 2. Excessive question marks
    # --------------------------------
    question_count = text.count("?")

    if question_count >= 3:
        signals.append(
            f"Excessive question marks detected ({question_count})"
        )
        score += 10

    # --------------------------------
    # 3. Excessive capitalization
    # --------------------------------
    letters = [char for char in text if char.isalpha()]

    if letters:
        uppercase_letters = [
            char for char in letters if char.isupper()
        ]

        uppercase_ratio = len(uppercase_letters) / len(letters)

        if uppercase_ratio >= 0.35 and len(letters) >= 20:
            signals.append(
                "Unusually high use of capital letters detected"
            )
            score += 15

    # --------------------------------
    # 4. Sensational language
    # --------------------------------
    lower_text = text.lower()

    matched_phrases = []

    for phrase in SENSATIONAL_PHRASES:
        if phrase in lower_text:
            matched_phrases.append(phrase)

    if matched_phrases:
        signals.append(
            "Sensational or emotionally charged language detected"
        )
        score += min(25, len(matched_phrases) * 8)

    # --------------------------------
    # 5. Very short content
    # --------------------------------
    word_count = len(text.split())

    if word_count < 15:
        signals.append(
            "Very limited content provided for analysis"
        )
        score += 10

    # --------------------------------
    # 6. Repeated punctuation
    # --------------------------------
    if re.search(r"[!?]{2,}", text):
        signals.append(
            "Repeated punctuation patterns detected"
        )
        score += 10

    # --------------------------------
    # Limit score
    # --------------------------------
    score = min(score, 100)

    # --------------------------------
    # Determine signal level
    # --------------------------------
    if score >= 50:
        signal_level = "High"
    elif score >= 25:
        signal_level = "Moderate"
    else:
        signal_level = "Low"

    return {
        "risk_score": score,
        "signals": signals,
        "signal_level": signal_level
    }