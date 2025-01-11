from flask import Flask, request, jsonify
from sentiment_analysis import get_sentiment
import langid

app = Flask(__name__)

# Root route
@app.route("/")
def home():
    return "Welcome to the Sentiment Analysis API! Use the /analyze-sentiment endpoint."

# Ignore favicon requests
@app.route("/favicon.ico")
def favicon():
    return "", 204

# Sentiment analysis route
@app.route("/analyze-sentiment", methods=["GET", "POST"])
def analyze_sentiment():
    """
    Analyze the sentiment of the provided text.

    Supports both GET and POST requests.
    - GET: Pass the text as a query parameter (`?text=your_text_here`).
    - POST: Pass the text in the form data or JSON body.
    """
    # Extract text from request
    if request.method == "GET":
        text = request.args.get("text")
    elif request.method == "POST":
        # Check if the request is JSON
        if request.is_json:
            data = request.get_json()
            text = data.get("text")
        else:
            text = request.form.get("text")
    else:
        return jsonify({"error": "Unsupported request method"}), 405

    # Validate text input
    if text is None or len(text.strip()) == 0:
        return jsonify({"error": "Text is required"}), 400

    # Detect the language of the input text
    lang, confidence = langid.classify(text)

    # Perform sentiment analysis and get the full sentiment result
    sentiment_result = get_sentiment(text, lang)

    # Return JSON response
    return jsonify({
        "text": text,
        "language": lang,
        "sentiment_classification": sentiment_result['sentiment_classification'],
        "sentiment_scores": sentiment_result['sentiment_scores']
    })

if __name__ == "__main__":
    app.run(debug=True)