import re


def analyze_claims(text):
    """
    Analyzes news text for claim-related signals.

    This is an NLP-based analysis layer.
    It does NOT determine whether a claim is factually true.
    """

    # -----------------------------
    # 1. Detect numerical claims
    # -----------------------------
    numerical_patterns = re.findall(
        r'\b\d+(?:\.\d+)?%?\b',
        text
    )

    # -----------------------------
    # 2. Detect years and dates
    # -----------------------------
    years = re.findall(
        r'\b(?:19|20)\d{2}\b',
        text
    )

    dates = re.findall(
        r'\b(?:'
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        r'|'
        r'\d{1,2}\s+'
        r'(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)'
        r'(?:\s+\d{4})?'
        r')\b',
        text,
        flags=re.IGNORECASE
    )

    # -----------------------------
    # 3. Detect URLs
    # -----------------------------
    urls = re.findall(
        r'https?://\S+|www\.\S+',
        text,
        flags=re.IGNORECASE
    )

    # -----------------------------
    # 4. Detect absolute language
    # -----------------------------
    absolute_words = [
        "always",
        "never",
        "everyone",
        "nobody",
        "completely",
        "definitely",
        "certainly",
        "100%",
        "guaranteed",
        "guarantee",
        "impossible",
        "undeniable",
        "proven"
    ]

    lower_text = text.lower()

    detected_absolute_words = [
        word for word in absolute_words
        if word in lower_text
    ]

    # -----------------------------
    # 5. Detect verification language
    # -----------------------------
    verification_phrases = [
        "according to",
        "official data",
        "study",
        "research",
        "report",
        "survey",
        "experts say",
        "experts believe",
        "scientists say",
        "officials said",
        "official statement",
        "published by"
    ]

    detected_verification_phrases = [
        phrase
        for phrase in verification_phrases
        if phrase in lower_text
    ]

    # -----------------------------
    # 6. Estimate number of claims
    # -----------------------------
    claim_indicators = [
        "said",
        "says",
        "reported",
        "reports",
        "found",
        "revealed",
        "according to",
        "confirmed",
        "claimed",
        "announced",
        "will",
        "has",
        "have",
        "is",
        "are"
    ]

    detected_claim_indicators = [
        phrase
        for phrase in claim_indicators
        if phrase in lower_text
    ]

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)

    meaningful_sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip().split()) >= 5
    ]

    claim_count = 0

    for sentence in meaningful_sentences:
        sentence_lower = sentence.lower()

        if any(
            indicator in sentence_lower
            for indicator in claim_indicators
        ):
            claim_count += 1

    # -----------------------------
    # 7. Calculate verification need
    # -----------------------------
    verification_score = 0

    if numerical_patterns:
        verification_score += 20

    if years or dates:
        verification_score += 10

    if detected_absolute_words:
        verification_score += 25

    if claim_count >= 2:
        verification_score += 20

    if urls:
        verification_score -= 10

    verification_score = max(
        0,
        min(100, verification_score)
    )

    if verification_score >= 60:
        verification_status = "High verification need"
    elif verification_score >= 30:
        verification_status = "Moderate verification need"
    else:
        verification_status = "Low verification need"

    # -----------------------------
    # 8. Return analysis
    # -----------------------------
    return {
        "claim_count": claim_count,
        "numerical_claims": len(numerical_patterns),
        "years_detected": len(years),
        "dates_detected": len(dates),
        "urls_detected": len(urls),
        "absolute_language": detected_absolute_words,
        "verification_phrases": detected_verification_phrases,
        "claim_indicators": detected_claim_indicators,
        "verification_score": verification_score,
        "verification_status": verification_status
    }