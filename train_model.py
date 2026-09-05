import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==============================
# LOAD WELFAKE DATASET
# ==============================

data = pd.read_csv("data/dataset.csv")

print("Dataset loaded successfully!")
print("Total articles:", len(data))


# ==============================
# CLEAN DATA
# ==============================

# Replace missing titles/text with empty strings
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Combine title + article text
data["content"] = data["title"] + " " + data["text"]

# Remove empty articles
data = data[data["content"].str.strip() != ""]

# Remove exact duplicate articles before splitting so no article appears in
# both the training and test partitions.
before_deduplication = len(data)
data = data.drop_duplicates(subset="content").reset_index(drop=True)
print("Exact duplicate articles removed:", before_deduplication - len(data))


# ==============================
# INPUT AND OUTPUT
# ==============================

X = data["content"]

# WELFake:
# 0 = Fake
# 1 = Real
y = data["label"]


# ==============================
# SPLIT DATASET
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))


# ==============================
# TF-IDF
# ==============================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")


# ==============================
# MODELS
# ==============================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes": MultinomialNB(),

    "SVM": LinearSVC()
}


# ==============================
# TRAIN AND COMPARE
# ==============================

results = {}

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")


for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=0
    )

    results[name] = accuracy

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")


# ==============================
# SELECT BEST MODEL
# ==============================

best_model_name = max(
    results,
    key=results.get
)

best_model = models[best_model_name]


print("\n==============================")
print("BEST MODEL:", best_model_name)
print("==============================")


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(
    best_model,
    "model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)


print("\nMODEL SAVED SUCCESSFULLY!")
print("Model       → model.pkl")
print("Vectorizer  → vectorizer.pkl")
print("\nTraining completed successfully!")