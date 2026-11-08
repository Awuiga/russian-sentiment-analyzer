
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Загрузка и очистка данных
df = pd.read_csv(
    "data/women-clothing-accessories.3-class.balanced.csv",
    sep="\t"
)

df["sentiment"] = df["sentiment"].replace({
    "neautral": "neutral"
})

label_counts = df.groupby("review")["sentiment"].nunique()
conflicts = label_counts[label_counts > 1].index

df = df[~df["review"].isin(conflicts)].copy()
df = df.drop_duplicates(subset=["review"]).copy()

print("Отзывов после очистки:", len(df))


# Разделение данных
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Train:", len(X_fit))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# Преобразование текстов в числа
vectorizer = TfidfVectorizer(max_features=10000)

X_fit_tfidf = vectorizer.fit_transform(X_fit)
X_val_tfidf = vectorizer.transform(X_val)


# Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_fit_tfidf, y_fit)

prediction = model.predict(X_val_tfidf)


# Оценка качества
accuracy = accuracy_score(y_val, prediction)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print(classification_report(y_val, prediction))

labels = ["negative", "neutral", "positive"]
matrix = confusion_matrix(y_val, prediction, labels=labels)

print("Матрица ошибок:")
print(matrix)


# Примеры неправильных предсказаний
results = pd.DataFrame({
    "review": X_val.to_numpy(),
    "actual": y_val.to_numpy(),
    "predicted": prediction
})

errors = results[
    (results["actual"] == "negative") &
    (results["predicted"] == "neutral")
]

print("\nNegative -> Neutral:", len(errors))
print(errors.head(5).to_string(index=False))

review = input('Введите отзыв: ')
review_tfidf = vectorizer.transform([review])
print(model.predict(review_tfidf)[0])

saved_data = {
    'model': model,
    'vectorizer': vectorizer
}

joblib.dump(saved_data, 'sentiment_model.joblib')
