from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_length=150):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/', methods=['GET'])
def home():
    return "Server is running"

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
        print(f"Received text: {text}")
        if text:
            summary = summarize_text(text)
            print(f"Generated summary: {summary}")
            return jsonify({'summary': summary}), 200
        else:
            return jsonify({'error': 'Text field is required'}), 400
    return jsonify({'error': 'Request must be JSON'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
