import re
from urllib.parse import urlparse


def analyze_source(url):
    """
    Analyzes a submitted news source URL.

    This is a source-structure analysis layer.
    It does NOT determine whether a website is trustworthy.
    """

    url = url.strip()

    if not url:
        return {
            "url_provided": False,
            "domain": None,
            "scheme": None,
            "https": False,
            "suspicious_url": False,
            "source_score": 0,
            "source_level": "No source provided",
            "signals": []
        }

    # --------------------------------------------------------
    # NORMALIZE URL
    # --------------------------------------------------------

    normalized_url = url

    if not re.match(r"^https?://", normalized_url, re.IGNORECASE):
        normalized_url = "https://" + normalized_url

    try:
        parsed = urlparse(normalized_url)

    except Exception:
        return {
            "url_provided": True,
            "domain": None,
            "scheme": None,
            "https": False,
            "suspicious_url": True,
            "source_score": 50,
            "source_level": "Invalid URL",
            "signals": [
                "The submitted source URL could not be parsed"
            ]
        }

    domain = parsed.netloc.lower()

    # Remove username/password and port information
    domain = domain.split("@")[-1]
    domain = domain.split(":")[0]

    scheme = parsed.scheme.lower()

    signals = []
    risk_score = 0


    # --------------------------------------------------------
    # HTTPS CHECK
    # --------------------------------------------------------

    if scheme == "https":

        https = True

    else:

        https = False

        signals.append(
            "Source does not use HTTPS"
        )

        risk_score += 15


    # --------------------------------------------------------
    # DOMAIN CHECK
    # --------------------------------------------------------

    if not domain:

        signals.append(
            "No valid domain was detected"
        )

        risk_score += 30


    # --------------------------------------------------------
    # SUSPICIOUS DOMAIN PATTERNS
    # --------------------------------------------------------

    suspicious_patterns = [

        r"free",
        r"viral",
        r"breaking",
        r"truth",
        r"exposed",
        r"secret",
        r"shocking",
        r"alert",
        r"news-now",
        r"news\d+",
        r"\d{4,}",
    ]

    matched_patterns = []

    for pattern in suspicious_patterns:

        if re.search(pattern, domain):

            matched_patterns.append(pattern)


    if matched_patterns:

        signals.append(
            "Domain contains patterns commonly associated "
            "with attention-driven or unusual naming"
        )

        risk_score += 15


    # --------------------------------------------------------
    # EXCESSIVE SUBDOMAINS
    # --------------------------------------------------------

    domain_parts = domain.split(".")

    if len(domain_parts) >= 4:

        signals.append(
            "Source uses an unusually deep subdomain structure"
        )

        risk_score += 10


    # --------------------------------------------------------
    # SUSPICIOUS CHARACTERS
    # --------------------------------------------------------

    if "_" in domain:

        signals.append(
            "Domain contains an underscore"
        )

        risk_score += 10


    # --------------------------------------------------------
    # IP ADDRESS AS DOMAIN
    # --------------------------------------------------------

    ip_pattern = r"^\d{1,3}(?:\.\d{1,3}){3}$"

    if re.match(ip_pattern, domain):

        signals.append(
            "Source uses an IP address instead of a domain name"
        )

        risk_score += 25


    # --------------------------------------------------------
    # VERY LONG DOMAIN
    # --------------------------------------------------------

    if len(domain) > 45:

        signals.append(
            "Domain name is unusually long"
        )

        risk_score += 10


    # --------------------------------------------------------
    # URL LENGTH
    # --------------------------------------------------------

    if len(url) > 180:

        signals.append(
            "URL is unusually long"
        )

        risk_score += 5


    # --------------------------------------------------------
    # QUERY PARAMETERS
    # --------------------------------------------------------

    if parsed.query:

        signals.append(
            "URL contains additional query parameters"
        )

        risk_score += 3


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    risk_score = max(
        0,
        min(100, risk_score)
    )


    # --------------------------------------------------------
    # SOURCE LEVEL
    # --------------------------------------------------------

    if risk_score >= 50:

        source_level = "High Risk Indicators"

    elif risk_score >= 25:

        source_level = "Moderate Risk Indicators"

    else:

        source_level = "Low Risk Indicators"


    # --------------------------------------------------------
    # SUSPICIOUS URL FLAG
    # --------------------------------------------------------

    suspicious_url = risk_score >= 25


    return {

        "url_provided": True,

        "domain": domain,

        "scheme": scheme,

        "https": https,

        "suspicious_url": suspicious_url,

        "source_score": risk_score,

        "source_level": source_level,

        "signals": signals
    }