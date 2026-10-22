
import json
import ipaddress
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishguard_model.joblib"
FEATURES_PATH = BASE_DIR / "phishguard_features.json"

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure",
    "banking", "signin", "password", "confirm", "wallet",
    "payment", "recover", "authenticate", "webscr"
]


# =========================================================
# 2. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    return model, feature_names


# =========================================================
# 3. URL FEATURE EXTRACTION
# Must match the notebook's training implementation
# =========================================================

def extract_url_features(url):
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "//" + url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    is_ip_address = 0

    try:
        ipaddress.ip_address(hostname)
        is_ip_address = 1
    except ValueError:
        pass

    digit_count = sum(char.isdigit() for char in url)
    special_chars = sum(not char.isalnum() for char in url)

    dot_count = hostname.count(".")
    subdomain_count = max(dot_count - 1, 0)

    if url:
        probabilities = [count / len(url) for count in Counter(url).values()]
        entropy = -sum(p * math.log2(p) for p in probabilities)
    else:
        entropy = 0.0

    suspicious_word_count = sum(url.lower().count(word) for word in SUSPICIOUS_WORDS)
    percent_encoding_count = len(re.findall(r"%[0-9a-fA-F]{2}", url))
    query_parameter_count = (len([item for item in query.split("&") if item]) if query else 0)

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": dot_count,
        "subdomain_count": subdomain_count,
        "digit_count": digit_count,
        "special_char_count": special_chars,
        "hyphen_count": url.count("-"),
        "at_symbol": int("@" in url),
        "www_subdomain": int(hostname.startswith("www.")),
        "https_scheme": int(
            parsed.scheme.lower() == "https"
        ),
        "ip_address_host": is_ip_address,
        "percent_encoding_count": percent_encoding_count,
        "query_parameter_count": query_parameter_count,
        "suspicious_word_count": suspicious_word_count,
        "entropy": entropy,
        "double_slash_in_path": int("//" in path),
    }


# =========================================================
# 4. VALIDATE INPUT
# =========================================================

def validate_url(url):
    url = str(url).strip()
    if not url:
        return False, "Please enter a URL."

    if len(url) > 10000:
        return False, "The URL is too long to process."

    # Do not silently interpret arbitrary text as a URL.
    candidate = url if "://" in url else "https://" + url

    try:
        parsed = urlparse(candidate)
        hostname = parsed.hostname

        if not hostname or "." not in hostname:
            return False, (
                "Enter a URL with a hostname, "
                "such as https://example.com")
        if any(char.isspace() for char in hostname):
            return False, "The hostname cannot contain spaces."

        # Accessing .port detects malformed port numbers.
        _ = parsed.port
        if parsed.scheme.lower() not in ("http", "https"):
            return False, "Only HTTP and HTTPS URLs are supported."

    except ValueError:
        return False, "The URL format is invalid."
    return True, ""
# =========================================================
# 5. PREDICTION AND MODEL-DERIVED RISK SCORE
# =========================================================

def predict_url(url, model, feature_names):
    valid, error_message = validate_url(url)

    if not valid:
        raise ValueError(error_message)

    url = str(url).strip()
    if "://" not in url:
        url = "https://" + url
    features = extract_url_features(url)
    input_df = pd.DataFrame([features],columns=feature_names)

    prediction = int(model.predict(input_df)[0])

    # Prefer the model's predicted phishing probability when
    # available. This score is NOT calibrated in this version.
    risk_score = None
    score_method = "Unavailable"

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        if 0 in classes:
            phishing_index = classes.index(0)
            risk_score = float(probabilities[phishing_index] * 100)
            score_method = "Uncalibrated model probability"

    # Fallback for models that expose a decision function only.
    elif hasattr(model, "decision_function"):
        decision = float(model.decision_function(input_df)[0])

        # For binary sklearn classifiers, a positive decision
        # value favors classes_[1]. Here label 0 means phishing.
        classes = list(model.classes_)

        if len(classes) == 2:
            phishing_direction = -1 if classes[1] == 1 else 1
            # Map the decision to a risk score between 0 and 100.
            z = max(-50,min(50, phishing_direction * decision))
            risk_score = 100 / (1 + math.exp(-z))
            score_method = "Uncalibrated decision score"

    result = ("Potentially Phishing" if prediction == 0 else "Potentially Legitimate")
    return {
        "url": url,
        "prediction": result,
        "label": prediction,
        "risk_score": risk_score,
        "score_method": score_method,
        "features": features
    }


# =========================================================
# 6. EXPLAIN URL INDICATORS
# These are heuristic indicators, not causal explanations.
# =========================================================

def explain_indicators(features):
    indicators = []

    if features["ip_address_host"]:
        indicators.append(("IP address used as hostname","An IP address is used instead of a conventional domain.","Review carefully"))

    if features["at_symbol"]:
        indicators.append(("At symbol in URL","The @ character can make parts of a URL confusing.","Review carefully"))

    if features["subdomain_count"] >= 3:
        indicators.append(("Many hostname levels","The hostname contains multiple dot-separated levels.","Review carefully"))

    if features["suspicious_word_count"] > 0:
        indicators.append(("Account-related keywords","Terms such as login, verify or password appear in the URL.","Context dependent"))

    if features["percent_encoding_count"] > 2:
        indicators.append(("Encoded characters","Several percent-encoded characters were detected.","Review carefully"))

    if features["digit_count"] >= 8:
        indicators.append(("Many digits","The URL contains a relatively large number of digits.","Context dependent"))

    if features["hyphen_count"] >= 3:
        indicators.append(("Multiple hyphens","Several hyphens occur in the URL.","Context dependent"))

    if features["url_length"] >= 100:
        indicators.append(("Long URL","The URL is unusually long by this prototype's rule.","Review carefully"))

    if not features["https_scheme"]:
        indicators.append(("No HTTPS scheme","The submitted URL does not use HTTPS.","Security concern"))

    if not indicators:
        indicators.append(("No listed heuristic triggered","The basic checks did not identify these particular patterns.","Not proof of safety"))

    return indicators


def get_risk_band(score):
    if score is None:
        return "Unavailable"

    if score >= 70:
        return "Higher model-indicated risk"
    elif score >= 40:
        return "Moderate model-indicated risk"
    else:
        return "Lower model-indicated risk"


# =========================================================
# 7. SESSION STATE
# =========================================================

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def add_to_history(result):
    record = {
        "Timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "URL": result["url"],
        "Prediction": result["prediction"],
        "Risk Score": (
            round(result["risk_score"], 2)
            if result["risk_score"] is not None
            else None
        )
    }

    st.session_state.scan_history.insert(0, record)


# =========================================================
# 8. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🛡️ PhishGuard AI")
st.sidebar.caption("Phishing awareness and URL analysis")

page = st.sidebar.radio(
    "Navigate",
    [
        "Introduction",
        "URL Analyzer",
        "Bulk Scanner",
        "Scan History",
        "About the Model"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application analyzes URL patterns. "
    "It does not visit websites or confirm malicious activity."
)


# =========================================================
# 9. LOAD MODEL
# =========================================================

try:
    model, feature_names = load_model()

    if len(feature_names) != 17:
        st.warning(
            "The saved feature list does not contain the "
            "expected 17 features. Check your notebook artifacts."
        )

except Exception as error:
    st.error(
        "Unable to load the trained model. Check that both "
        "saved model files are located beside app.py."
    )
    st.exception(error)
    st.stop()


# =========================================================
# 10. INTRODUCTION PAGE
# =========================================================

if page == "Introduction":

    st.title("PhishGuard - Phishing Awareness and URL Analysis")

    st.subheader(
        "Understand phishing. Analyze URLs. Make informed decisions."
    )

    st.write(
        "PhishGuard AI is an educational cybersecurity prototype "
        "that uses machine learning to classify URLs based on "
        "characteristics learned from a labeled dataset."
    )

    st.warning(
        "A URL prediction is not a guarantee of safety. "
        "Never share passwords, OTPs or banking details with "
        "a site solely because this tool labels it legitimate."
    )

    st.markdown("---")
    st.header("What is phishing?")

    st.write(
        "Phishing is a form of social engineering in which "
        "attackers impersonate trusted people or organizations "
        "to trick users into revealing sensitive information, "
        "opening malicious links or taking unsafe actions."
    )

    
    st.markdown("---")
    st.header("How does PhishGuard AI work?")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("1. Extract")
        st.write(
            "Extract numerical characteristics from the submitted URL."
        )

    with c2:
        st.subheader("2. Predict")
        st.write(
            "Pass those characteristics into the saved ML pipeline."
        )

    with c3:
        st.subheader("3. Explain")
        st.write(
            "Present the classification, model-derived score "
            "and relevant URL indicators."
        )

    st.markdown("---")
    st.header("Example URL patterns")

    example_data = pd.DataFrame([
        {
            "Pattern": "Unexpected account verification link",
            "Why review it?": "It may be part of a credential-harvesting attempt."
        },
        {
            "Pattern": "Hostname uses an IP address",
            "Why review it?": "The destination may be less recognizable."
        },
        {
            "Pattern": "Familiar brand name inside an unrelated domain",
            "Why review it?": "The hostname may imitate a trusted service."
        },
        {
            "Pattern": "HTTPS connection",
            "Why review it?": "HTTPS encrypts the connection but does not prove legitimacy."
        }
    ])

    st.dataframe(
        example_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These patterns are examples for awareness, not definitive "
        "rules for identifying phishing."
    )


# =========================================================
# 11. URL ANALYZER PAGE
# =========================================================

elif page == "URL Analyzer":

    st.title("URL Analyzer")
    st.write(
        "Submit a URL to examine its characteristics and receive "
        "a machine-learning assessment."
    )

    with st.form("single_url_form"):
        url_input = st.text_input(
            "URL to analyze",
            placeholder="https://example.com/login"
        )

        analyze_button = st.form_submit_button(
            "Analyze URL",
            type="primary"
        )

    if analyze_button:
        try:
            result = predict_url(
                url_input,
                model,
                feature_names
            )

            st.session_state.last_result = result
            add_to_history(result)

        except ValueError as error:
            st.error(str(error))

    result = st.session_state.last_result

    if result:
        st.markdown("---")
        st.header("Analysis result")

        st.code(result["url"], language=None)

        if result["label"] == 0:
            st.error("Potentially Phishing")
            st.write(
                "The trained model classified this URL as phishing. "
                "Avoid entering credentials unless you independently "
                "verify the destination."
            )
        else:
            st.success("Potentially Legitimate")
            st.write(
                "The model classified this URL as legitimate. "
                "This does not prove the website is safe."
            )

        score = result["risk_score"]

        st.subheader("Model-derived risk indicator")

        if score is not None:
            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Phishing risk score",
                    f"{score:.1f}/100"
                )

            with c2:
                st.metric(
                    "Risk band",
                    get_risk_band(score)
                )

            st.progress(
                max(0.0, min(1.0, score / 100))
            )

            st.caption(
                f"Score source: {result['score_method']}. "
                "This score is not calibrated and should not be "
                "interpreted as a verified real-world probability."
            )
        else:
            st.info(
                "The saved model does not expose a supported score. "
                "Only the model's classification is displayed."
            )

        st.markdown("---")
        st.subheader("URL structure")

        parsed = urlparse(result["url"])

        c1, c2 = st.columns(2)

        with c1:
            st.write("**Scheme:**", parsed.scheme or "Not specified")
            st.write("**Hostname:**", parsed.hostname or "Unavailable")
            st.write("**Path:**", parsed.path or "/")

        with c2:
            st.write(
                "**Query string present:**",
                "Yes" if parsed.query else "No"
            )
            st.write(
                "**Fragment present:**",
                "Yes" if parsed.fragment else "No"
            )
            st.write(
                "**URL length:**",
                len(result["url"])
            )

        st.markdown("---")
        st.subheader("URL indicators")

        indicators = explain_indicators(result["features"])

        indicators_df = pd.DataFrame(
            indicators,
            columns=["Indicator", "Explanation", "Interpretation"]
        )

        st.dataframe(
            indicators_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These are rule-based observations about URL patterns. "
            "They are not guaranteed explanations of the model's "
            "internal decision."
        )

        with st.expander("View all extracted ML features"):
            feature_df = pd.DataFrame(
                list(result["features"].items()),
                columns=["Feature", "Value"]
            )

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )

        report_df = pd.DataFrame([{
            "URL": result["url"],
            "Prediction": result["prediction"],
            "Phishing Risk Score": score,
            "Risk Score Type": result["score_method"]
        }])

        st.download_button(
            "Download this scan as CSV",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="phishguard_single_scan.csv",
            mime="text/csv"
        )


# =========================================================
# 12. BULK SCANNER PAGE
# =========================================================

elif page == "Bulk Scanner":

    st.title("Bulk URL Scanner")

    st.write(
        "Upload a CSV file containing a column named URL. "
        "The application will analyze each URL without visiting it."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if "URL" not in uploaded_df.columns:
                st.error(
                    "Your CSV must contain a column named URL."
                )

            elif len(uploaded_df) == 0:
                st.warning("The uploaded CSV contains no rows.")

            elif len(uploaded_df) > 10000:
                st.error(
                    "Please upload no more than 10,000 URLs per file."
                )

            else:
                st.write(
                    f"URLs loaded: **{len(uploaded_df):,}**"
                )

                if st.button(
                    "Analyze all URLs",
                    type="primary"
                ):
                    records = []
                    progress = st.progress(0)
                    status = st.empty()

                    total = len(uploaded_df)

                    for index, raw_url in enumerate(
                        uploaded_df["URL"]
                    ):
                        try:
                            result = predict_url(
                                raw_url,
                                model,
                                feature_names
                            )

                            score = result["risk_score"]

                            records.append({
                                "URL": result["url"],
                                "Prediction": result["prediction"],
                                "Phishing Risk Score": (
                                    round(score, 2)
                                    if score is not None
                                    else None
                                ),
                                "Status": "Processed"
                            })

                            add_to_history(result)

                        except Exception as error:
                            records.append({
                                "URL": str(raw_url),
                                "Prediction": "Not analyzed",
                                "Phishing Risk Score": None,
                                "Status": str(error)
                            })

                        progress.progress(
                            (index + 1) / total
                        )

                        status.write(
                            f"Analyzed {index + 1:,} of {total:,} URLs"
                        )

                    results_df = pd.DataFrame(records)

                    st.session_state["last_batch_results"] = (
                        results_df
                    )

                    status.success("Batch analysis completed.")

        except Exception as error:
            st.error("Unable to read the uploaded CSV.")
            st.exception(error)

    results_df = st.session_state.get("last_batch_results")

    if results_df is not None:
        st.markdown("---")
        st.header("Batch results")

        total = len(results_df)

        phishing_count = (
            results_df["Prediction"] == "Potentially Phishing"
        ).sum()

        legitimate_count = (
            results_df["Prediction"] == "Potentially Legitimate"
        ).sum()

        failed_count = (
            results_df["Prediction"] == "Not analyzed"
        ).sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total URLs", f"{total:,}")
        c2.metric("Potentially phishing", f"{phishing_count:,}")
        c3.metric("Potentially legitimate", f"{legitimate_count:,}")
        c4.metric("Not analyzed", f"{failed_count:,}")

        filter_option = st.selectbox(
            "Filter results",
            [
                "All",
                "Potentially Phishing",
                "Potentially Legitimate",
                "Not analyzed"
            ]
        )

        if filter_option == "All":
            filtered_df = results_df
        else:
            filtered_df = results_df[
                results_df["Prediction"] == filter_option
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download batch report",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="phishguard_batch_report.csv",
            mime="text/csv"
        )


# =========================================================
# 13. SCAN HISTORY PAGE
# =========================================================

elif page == "Scan History":

    st.title("Scan History")

    st.write(
        "Review scans recorded during the current Streamlit session."
    )

    history = st.session_state.scan_history

    if not history:
        st.info(
            "No scans have been recorded yet. "
            "Analyze a URL or run a bulk scan first."
        )

    else:
        history_df = pd.DataFrame(history)

        c1, c2 = st.columns(2)

        c1.metric(
            "Total recorded scans",
            f"{len(history_df):,}"
        )

        c2.metric(
            "Potentially phishing",
            f"{(history_df['Prediction'] == 'Potentially Phishing').sum():,}"
        )

        search_text = st.text_input(
            "Search URL history",
            placeholder="Enter a domain or keyword"
        )

        if search_text:
            filtered_history = history_df[
                history_df["URL"].str.contains(
                    search_text,
                    case=False,
                    na=False,
                    regex=False
                )
            ]
        else:
            filtered_history = history_df

        st.dataframe(
            filtered_history,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export scan history",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="phishguard_scan_history.csv",
            mime="text/csv"
        )

        if st.button("Clear scan history"):
            st.session_state.scan_history = []
            st.rerun()


# =========================================================
# 14. ABOUT THE MODEL PAGE
# =========================================================

elif page == "About the Model":

    st.title("About the Machine Learning Model")

    st.write(
        "PhishGuard uses a trained scikit-learn pipeline "
        "to classify URLs based on numerical features extracted "
        "from their text."
    )

    st.header("Current model")

    st.write(
        "This application loads the tuned model saved by the "
        "notebook. It does not retrain the model when the dashboard starts."
    )

    st.write("**Model artifact:**", MODEL_PATH.name)
    st.write("**Number of input features:**", len(feature_names))

    st.header("Features used")

    st.markdown(
        """
        - URL length
        - Hostname length
        - Path length
        - Query length
        - Dot count and subdomain count
        - Digit count and special-character count
        - Hyphen count
        - Presence of an @ symbol
        - HTTPS scheme
        - IP address used as hostname
        - Percent-encoded character count
        - Query parameter count
        - Suspicious keyword count
        - Character entropy
        - Double slash in the path
        """
    )


    st.header("Current limitations")

    st.markdown(
        """
        - The model analyzes URL text only.
        - It does not fetch webpage content or check domain reputation.
        - The risk score is not calibrated in this version.
        - URL patterns can be misleading, and legitimate sites can
          contain suspicious-looking strings.
        - The scan history is stored in session state and is temporary.
        """
    )

    st.info(
        "The model's measured performance should be reported using "
        "the results saved from your actual held-out test set. "
        "Do not treat example metrics as experimental results."
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "PhishGuard | Academic machine-learning project | "
    "URL-based assessment, not a security guarantee."
)
