import pandas as pd
import joblib
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/dataset.csv")

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["content"] = (
    df["title"].astype(str) + " " + df["text"].astype(str)
).str.strip()

df = df[df["content"].str.len() > 0]

# Remove exact duplicate articles
df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)

print("Unique articles:", len(df))


# --------------------------------------------------
# 2. Recreate the same test split
# --------------------------------------------------

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Use 1,000 articles for a practical Transformer benchmark
X_test = X_test.iloc[:1000]
y_test = y_test.iloc[:1000]

print("Benchmark articles:", len(X_test))


# --------------------------------------------------
# 3. Test existing SVM
# --------------------------------------------------

print("\nLoading existing SVM...")

svm = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

X_test_tfidf = vectorizer.transform(X_test)

svm_predictions = svm.predict(X_test_tfidf)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\n==============================")
print("LINEAR SVM RESULTS")
print("==============================")
print("Accuracy:", round(svm_accuracy * 100, 2), "%")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        svm_predictions,
        target_names=["Fake", "Real"]
    )
)
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))


# --------------------------------------------------
# 4. Load Transformer
# --------------------------------------------------

print("\nLoading Transformer...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(
    "GhostFaith/distilbert-fakenews"
)

transformer = AutoModelForSequenceClassification.from_pretrained(
    "GhostFaith/distilbert-fakenews"
).to(device)

transformer.eval()


# --------------------------------------------------
# 5. Transformer predictions
# --------------------------------------------------

transformer_predictions = []

texts = X_test.tolist()

print("\nRunning Transformer predictions...")

batch_size = 16

with torch.no_grad():

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded.pop("token_type_ids", None)

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        outputs = transformer(**encoded)

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        ).cpu().tolist()

        transformer_predictions.extend(predictions)

        print(
            f"Processed {min(i + batch_size, len(texts))}/{len(texts)}"
        )


# --------------------------------------------------
# 6. Transformer results
# --------------------------------------------------

transformer_accuracy = accuracy_score(
    y_test,
    transformer_predictions
)

print("\n==============================")
print("TRANSFORMER RESULTS")
print("==============================")
print(
    "Accuracy:",
    round(transformer_accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        transformer_predictions,
        target_names=["Fake", "Real"]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        transformer_predictions
    )
)


# --------------------------------------------------
# 7. Final comparison
# --------------------------------------------------

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print(
    "Linear SVM:",
    round(svm_accuracy * 100, 2),
    "%"
)

print(
    "Transformer:",
    round(transformer_accuracy * 100, 2),
    "%"
)

difference = (
    transformer_accuracy - svm_accuracy
) * 100

print(
    "Difference:",
    round(difference, 2),
    "percentage points"
)

if difference > 0:
    print("\nTransformer performs better on this benchmark.")

elif difference < 0:
    print("\nLinear SVM performs better on this benchmark.")

else:
    print("\nBoth models have the same accuracy.")