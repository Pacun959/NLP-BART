from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import keras
import os
import time
import keras_nlp
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ["KERAS_BACKEND"] = "jax" 
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 128


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
def generate_text(model, input_text, max_length=200, print_time_taken=False):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    end = time.time()
    print(f"Total Time Elapsed: {end - start:.2f}s")
    return output
preprocessor =  keras_nlp.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)
model = keras_nlp.models.BartSeq2SeqLM.from_preset(
    "bart_base_en", preprocessor=preprocessor
)
model_path = "./model/bart_model.keras"
model = keras.models.load_model(model_path)
def generate_text(model, input_text, max_length=200, print_time_taken=False):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    end = time.time()
    print(f"Total Time Elapsed: {end - start:.2f}s")
    return output

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data['text']
    word = data['output_sentences']
    # Tóm tắt văn bản bằng Bert-extractive-summarizer
    generated_summaries = generate_text(
    model,
    text,
    max_length=word,
    print_time_taken=True,
)

    return jsonify({
        'input_text': text,
        'summary_bert_extractive': generated_summaries,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)