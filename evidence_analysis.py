import re
import requests

from bs4 import BeautifulSoup
from urllib.parse import urlparse

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# ==========================================================
# SEMANTIC MODEL
# ==========================================================

SEMANTIC_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

semantic_model = None


def get_semantic_model():
    """
    Loads the Sentence Transformer model only when required.
    """

    global semantic_model

    if semantic_model is None:

        print(
            "Loading semantic evidence model..."
        )

        semantic_model = SentenceTransformer(
            SEMANTIC_MODEL_NAME
        )

        print(
            "Semantic evidence model loaded successfully!"
        )

    return semantic_model


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# SOURCE TEXT EXTRACTION
# ==========================================================

def retrieve_source(source_url):

    if not source_url:

        return {
            "success": False,
            "title": None,
            "text": "",
            "error": "No source URL provided."
        }


    url = source_url.strip()


    if not re.match(
        r"^https?://",
        url,
        flags=re.IGNORECASE
    ):

        url = "https://" + url


    try:

        parsed = urlparse(url)


        if parsed.scheme not in [
            "http",
            "https"
        ]:

            return {
                "success": False,
                "title": None,
                "text": "",
                "error": (
                    "Unsupported URL scheme."
                )
            }


        response = requests.get(

            url,

            timeout=10,

            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Macintosh; Intel Mac OS X) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120 Safari/537.36"
                )
            }

        )


        response.raise_for_status()


    except Exception as error:

        return {
            "success": False,
            "title": None,
            "text": "",
            "error": (
                f"{type(error).__name__}"
            )
        }


    try:

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )


        # Remove non-content elements

        for element in soup([

            "script",
            "style",
            "noscript",
            "svg",
            "iframe",
            "nav",
            "footer",
            "header"

        ]):

            element.decompose()


        title = None

        if soup.title:

            title = soup.title.get_text(
                " ",
                strip=True
            )


        source_text = soup.get_text(
            " ",
            strip=True
        )


        source_text = clean_text(
            source_text
        )


    except Exception as error:

        return {
            "success": False,
            "title": None,
            "text": "",
            "error": (
                f"{type(error).__name__}"
            )
        }


    if not source_text:

        return {
            "success": False,
            "title": title,
            "text": "",
            "error": (
                "No readable text extracted."
            )
        }


    return {
        "success": True,
        "title": title,
        "text": source_text,
        "error": None
    }


# ==========================================================
# TEXT CHUNKING
# ==========================================================

def create_chunks(
    text,
    words_per_chunk=180,
    max_chunks=12
):

    words = text.split()

    chunks = []


    for i in range(
        0,
        len(words),
        words_per_chunk
    ):

        chunk = " ".join(
            words[
                i:i + words_per_chunk
            ]
        )

        if chunk:

            chunks.append(chunk)


        if len(chunks) >= max_chunks:

            break


    return chunks


# ==========================================================
# KEYWORD OVERLAP
# ==========================================================

def calculate_keyword_overlap(
    news_text,
    source_text
):

    news_words = re.findall(
        r"\b[a-zA-Z]{4,}\b",
        news_text.lower()
    )


    source_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            source_text.lower()
        )
    )


    stop_words = {

        "this",
        "that",
        "with",
        "from",
        "after",
        "before",
        "about",
        "have",
        "been",
        "were",
        "they",
        "their",
        "there",
        "which",
        "would",
        "could",
        "should",
        "into",
        "than",
        "then",
        "also",
        "more",
        "some",
        "such",
        "said",
        "says",
        "news",
        "article",
        "according",
        "reported"

    }


    important_words = []


    for word in news_words:

        if word not in stop_words:

            important_words.append(word)


    important_words = list(
        dict.fromkeys(
            important_words
        )
    )


    matched_terms = [

        word

        for word in important_words

        if word in source_words

    ]


    if important_words:

        score = (

            len(matched_terms)

            /

            len(important_words)

        ) * 100

    else:

        score = 0


    return (
        round(
            min(score, 100),
            1
        ),
        matched_terms
    )


# ==========================================================
# SEMANTIC SIMILARITY
# ==========================================================

def calculate_semantic_similarity(
    news_text,
    source_text
):

    try:

        model = get_semantic_model()


        chunks = create_chunks(
            source_text
        )


        if not chunks:

            return 0.0


        news_embedding = model.encode(

            news_text,

            convert_to_tensor=True,

            normalize_embeddings=True

        )


        chunk_embeddings = model.encode(

            chunks,

            convert_to_tensor=True,

            normalize_embeddings=True

        )


        similarities = cos_sim(

            news_embedding,

            chunk_embeddings

        )[0]


        best_score = float(
            similarities.max().item()
        )


        # Convert cosine similarity
        # from approximately [-1, 1]
        # into a readable 0-100 score.

        semantic_score = (
            (best_score + 1)
            / 2
        ) * 100


        semantic_score = max(
            0,
            min(
                semantic_score,
                100
            )
        )


        return round(
            semantic_score,
            1
        )


    except Exception as error:

        print(
            "Semantic similarity error:",
            type(error).__name__
        )

        return 0.0


# ==========================================================
# FACT-CHECKING LANGUAGE
# ==========================================================

def detect_contradiction_language(
    source_text
):

    lower_source = (
        source_text.lower()
    )


    contradiction_phrases = [

        "false claim",
        "false",
        "misleading",
        "no evidence",
        "incorrect",
        "fact check",
        "fact-check",
        "debunked",
        "untrue",
        "not true",
        "fake news",
        "fabricated",
        "wrong claim",
        "baseless claim",
        "old video",
        "old image",
        "misrepresented",
        "not related",
        "unrelated"

    ]


    matches = [

        phrase

        for phrase in contradiction_phrases

        if phrase in lower_source

    ]


    return matches


# ==========================================================
# MAIN EVIDENCE ANALYSIS
# ==========================================================

def analyze_evidence(
    news_text,
    source_url
):

    result = {

        "evidence_available": False,

        "source_title": None,

        "source_text_length": 0,

        "matched_terms": [],

        "match_score": 0,

        "semantic_score": 0,

        "combined_score": 0,

        "evidence_status":
            "No evidence retrieved",

        "evidence_strength":
            "Unavailable",

        "signals": []

    }


    # ======================================================
    # RETRIEVE SOURCE
    # ======================================================

    retrieved = retrieve_source(
        source_url
    )


    if not retrieved["success"]:

        result["signals"].append(
            "The source webpage could not be retrieved."
        )

        result["signals"].append(
            "Retrieval detail: "
            +
            str(
                retrieved["error"]
            )
        )

        return result


    source_text = retrieved["text"]


    result[
        "evidence_available"
    ] = True


    result[
        "source_title"
    ] = retrieved[
        "title"
    ]


    result[
        "source_text_length"
    ] = len(
        source_text
    )


    # ======================================================
    # KEYWORD MATCHING
    # ======================================================

    keyword_score, matched_terms = (
        calculate_keyword_overlap(

            news_text,

            source_text

        )
    )


    result[
        "matched_terms"
    ] = matched_terms[:20]


    result[
        "match_score"
    ] = keyword_score


    # ======================================================
    # SEMANTIC MATCHING
    # ======================================================

    semantic_score = (
        calculate_semantic_similarity(

            news_text,

            source_text

        )
    )


    result[
        "semantic_score"
    ] = semantic_score


    # ======================================================
    # COMBINED EVIDENCE SCORE
    # ======================================================

    combined_score = (

        keyword_score * 0.35

        +

        semantic_score * 0.65

    )


    result[
        "combined_score"
    ] = round(
        combined_score,
        1
    )


    # ======================================================
    # FACT-CHECK LANGUAGE
    # ======================================================

    contradiction_matches = (
        detect_contradiction_language(
            source_text
        )
    )


    # ======================================================
    # EVIDENCE INTERPRETATION
    # ======================================================

    if contradiction_matches:

        result[
            "evidence_status"
        ] = (
            "Potential contradiction detected"
        )


        result[
            "evidence_strength"
        ] = "Strong"


        result["signals"].append(

            "The source contains explicit "
            "fact-checking or contradiction-related "
            "language."

        )


    elif combined_score >= 75:

        result[
            "evidence_status"
        ] = (
            "Strong source relevance"
        )


        result[
            "evidence_strength"
        ] = "Strong"


        result["signals"].append(

            "The submitted claim shows strong "
            "semantic and textual relevance "
            "to the retrieved source."

        )


    elif combined_score >= 50:

        result[
            "evidence_status"
        ] = (
            "Moderate source relevance"
        )


        result[
            "evidence_strength"
        ] = "Moderate"


        result["signals"].append(

            "The submitted claim shows moderate "
            "semantic relevance to the retrieved source."

        )


    elif combined_score >= 25:

        result[
            "evidence_status"
        ] = (
            "Limited source relevance"
        )


        result[
            "evidence_strength"
        ] = "Limited"


        result["signals"].append(

            "Some relationship was detected, "
            "but the evidence relevance is limited."

        )


    else:

        result[
            "evidence_status"
        ] = (
            "Weak source relevance"
        )


        result[
            "evidence_strength"
        ] = "Weak"


        result["signals"].append(

            "Little semantic or textual overlap "
            "was found between the claim and source."

        )


    # ======================================================
    # ADD SCORE SIGNAL
    # ======================================================

    result["signals"].append(

        "Semantic evidence similarity: "
        +
        str(
            semantic_score
        )
        +
        "%."

    )


    return result