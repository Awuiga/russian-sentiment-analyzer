import joblib
saved_data = joblib.load("sentiment_model.joblib")

vectorizer = saved_data['vectorizer']
model = saved_data['model']

while True:
    review = input('Введите свой отзыв:')
    if review.lower() == "выход":
        break

    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    print(prediction)
