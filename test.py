import joblib

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("Enter a review (type 'exit' to stop):")

while True:
    review = input("> ")

    if review.lower() == "exit":
        break

    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]

    if prediction == 1:
        print("✅ Genuine Review")
    else:
        print("❌ Fake Review")
