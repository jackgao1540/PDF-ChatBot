from llama_cpp import Llama 
import os
import os.path
import sys
from sentence_transformers import SentenceTransformer
from torch.hub import _get_torch_home
import datetime
import numpy as np
import faiss
from datetime import datetime


def process_input(question, remove_limits=False):
    if remove_limits:
        return LLM(question, max_tokens=0)
    return LLM(question)

def find_similar_for_id(id, embeddings, index, k=10):
    embedding = embeddings[id]
    _, I = index.search(np.array([embedding]), k)
    return I[0]

def find_similar_for_embedding(embedding, index, k=10):
    _, I = index.search(np.array([embedding]), k)
    return I[0]

def formulate_prompt(similar_ids, question, paragraphs):
    relevant_paragraphs = [paragraphs[id] for id in similar_ids]
    paragraphs_text= ', '.join(relevant_paragraphs)
    prompt = f"Given the following paragraphs of information, {paragraphs_text}, answer the following question to the best of your ability: {question}"
    return prompt

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: promptModels.py <model_number> [remove_limits]")
        sys.exit(1)
    model_number = sys.argv[1]
    remove_limits = False
    if not model_number.isdigit():
        print("Please provide a valid integer for model number.")
        sys.exit(1)
    if len(sys.argv) == 3:
        remove_limits = sys.argv[2] == "1"
    file_name = f"llama_{model_number}.bin"
    if not os.path.exists(file_name):
        print(f"{file_name} does not exist.")
        sys.exit(1)
    global LLM 
    print("Loading the LLM...")
    if remove_limits:
        LLM = Llama(model_path=str(file_name), n_ctx=2048)
    else:
        LLM = Llama(model_path=str(file_name))
    print("Loading Sentence Transformer model...")
    embeddings_model = SentenceTransformer("sentence-transformers/gtr-t5-large")

    embeddings_list=[]
    paragraphs_list=[]

    embedding_files = [f for f in os.listdir() if f.startswith("embeddings") and f.endswith(".npy")]
    embedding_files.sort(key=lambda x: int(x.split("embeddings_")[1].split(".npy")[0]))


    if not embedding_files:
        print("Error: No embedding files found!")
        exit(1)
    
    for embeddings_filename in embedding_files:
        print(f"Loading {embeddings_filename}...")
        embeddings = np.load(embeddings_filename)
        embeddings_list.append(embeddings)
        data_index = int(embeddings_filename.split("embeddings_")[1].split(".npy")[0])
        data_filename = f"data{data_index}.txt"
        with open(data_filename, 'r', encoding='utf-8') as f:
            paragraphs = f.read().split("\n\n")
        paragraphs_list.append(paragraphs)
    embeddings = [embedding for sublist in embeddings_list for embedding in sublist]
    paragraphs = [p for sublist in paragraphs_list for p in sublist]

    questions = []
    answers = []
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    while True:
        question = input("Enter your question (or 'exit to quit): ")
        if question.strip().lower() == 'exit':
            break
        question_embedding = embeddings_model.encode(question)
        indices=find_similar_for_embedding(question_embedding, index, 10)
        similar_ids=[]
        for idx in indices:
            similar_ids.append(idx)
            if idx + 1 < len(embeddings):
                similar_ids.append(idx+1)
            if idx + 2 < len(embeddings):
                similar_ids.append(idx+2)
            
        questions.append([question, question_embedding])
        prompt = formulate_prompt(similar_ids, question, paragraphs)
        print("Updated prompt: " + prompt)
        print("Model is processing...")
        answer = process_input(prompt, remove_limits)["choices"][0]["text"]
        print(f"Model's answer: {answer}")
        answers.append(answer)
    
    filename = f"prompts_and_answers_{datetime.now().strftime('%Y-%M-%D %H:%M:%S')}.txt"
    with open(filename, 'w') as file:
        for q, a in questions, answers:
            file.write("Question: " + q + "\n")
            file.write("Answer: " + a + "\n")

if __name__ == "__main__":
    main()
