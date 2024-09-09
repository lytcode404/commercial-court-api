from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from flask_cors import CORS  # Import CORS
import openai
import numpy as np
import faiss
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)


# Load OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("The environment variable OPENAI_API_KEY is not set.")

# Set OpenAI API key
openai.api_key = openai_api_key

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Load the DataFrame
df = pd.read_json('new_update.json')

# # Flatten section-content into a single string for each entry
# df['section-content-text'] = df['section-content'].apply(
#     lambda x: ' '.join(x.get('Content', [])) if isinstance(x, dict) else ''
# )
# df = df.drop(columns=['section-content'])

# Convert section-content-text into Document objects
documents = [Document(page_content=text)
             for text in df['section-content-text'].tolist()]

# Initialize FAISS index
faiss_index = FAISS.from_documents(documents, embeddings)


@app.route('/find_similar_sections', methods=['POST'])
def find_similar_sections():
    # Get the query from the request
    query = request.json.get('query', '')

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    # Embed the query
    query_embedding = embeddings.embed_query(query)

    # Ensure query_embedding is a 2D numpy array
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Search the FAISS index for the nearest neighbors
    distances, indices = faiss_index.index.search(query_embedding, 3)

    # Retrieve the corresponding rows from the DataFrame
    similar_sections = df.iloc[indices[0]]

    # Prepare the response
    result = similar_sections[['chapter', 'Section']].to_dict(orient='records')

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
