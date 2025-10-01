from flask import Flask, request, jsonify
import joblib
from preprocessing import SpacyPreprocessor  # 👈 important

# Load the saved pipeline
model = joblib.load("sentiment_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tweet = data.get("text")

    if not tweet:
        return jsonify({"error": "No tweet text provided"}), 400

    prediction = model.predict([tweet])[0]
    sentiment = "positive" if prediction == 1 else "negative"

    return jsonify({
        "tweet": tweet,
        "prediction": int(prediction),
        "sentiment": sentiment
    })


if __name__ == "__main__":
    app.run(debug=True)

