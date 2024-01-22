import sys
print(sys.executable)

from flask import Flask, request, render_template, jsonify
import os
from io import BytesIO
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pdfminer.high_level import extract_text
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#  smaller, faster model: 
# model_id = "amazon/MistralLite"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

def extract_paragraphs_from_pdf(pdf_path):
    txt=extract_text(pdf_path)
    paragraphs=[p for p in txt.split("\n\n") if p.strip()]
    return paragraphs

def save_paragraphs_to_txt(paragraphs, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in paragraphs:
            if len(p) > 1:
                f.write(p + "\n\n")

# pdf_paths=["HP.pdf"]

# for idx, pdf_path in enumerate(pdf_paths, 1):
#     paragraphs = extract_paragraphs_from_pdf(pdf_path)
#     for p in paragraphs:
#         print(p)
#         print('-' * 50)

#     output_file_name = f"data{idx}.txt"
#     save_paragraphs_to_txt(paragraphs, output_file_name)
#     print(f"Saved paragraphs from {pdf_path} to {output_file_name}")


# Initialize Sentence Transformer
embeddings_model = SentenceTransformer("sentence-transformers/gtr-t5-large")

# Initialize FAISS index (assuming dimension of embeddings is 768, change if different)
dimension = 768
index = faiss.IndexFlatL2(dimension)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    question = request.form.get('question')
    if uploaded_file and question:
        # Read the PDF file
        paragraphs = extract_paragraphs_from_pdf(BytesIO(uploaded_file.read()))
        
        # Generate embeddings for paragraphs
        embeddings = embeddings_model.encode(paragraphs)
        
        # Add embeddings to FAISS index
        index.reset()  # Reset index for new file
        index.add(np.array(embeddings))

        # Find similar paragraphs for the question
        question_embedding = embeddings_model.encode([question])
        _, indices = index.search(np.array(question_embedding), 10)

        # Retrieve and concatenate similar paragraphs
        similar_paragraphs = [paragraphs[i] for i in indices[0]]
        context = " ".join(similar_paragraphs)

        # Placeholder for your LLM's response generation
        # You need to replace this with the actual call to your model
        answer = generate_response_with_llm(context, question)

        return jsonify({'response': answer})
    
    return jsonify({'error': 'Please upload a file and ask a question'})

def generate_response_with_llm(context, question):
    # Placeholder function for LLM integration
    # Replace with actual model call

    text = context + question
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True)
