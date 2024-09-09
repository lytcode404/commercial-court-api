from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
import openai
import numpy as np
import faiss
import pandas as pd
import os

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("The environment variable OPENAI_API_KEY is not set.")


# Set your OpenAI API key
openai.api_key = openai_api_key


# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)


df = pd.read_json('new_update.json')
# # Flatten the section-content into a single string for each entry
# df['section-content-text'] = df['section-content'].apply(
#     lambda x: ' '.join(x.get('Content', [])) if isinstance(x, dict) else ''
# )
# df = df.drop(columns=['section-content'])

# Convert section-content-text into Document objects
documents = [Document(page_content=text)
             for text in df['section-content-text'].tolist()]

# Initialize FAISS index
faiss_index = FAISS.from_documents(documents, embeddings)

# Save the FAISS index using FAISS library directly
# faiss.write_index(faiss_index.index, "faiss_index.index")

# Function to find similar sections based on a query


def find_similar_sections(query, faiss_index, df, top_k=3):
    # Embed the query
    query_embedding = embeddings.embed_query(query)

    # Ensure query_embedding is a 2D numpy array
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Search the FAISS index for the nearest neighbors
    distances, indices = faiss_index.index.search(query_embedding, top_k)

    # Retrieve the corresponding rows from the DataFrame
    similar_sections = df.iloc[indices[0]]

    return similar_sections


# Example query
query = '''
A large international company based in India, "GlobalTech Ltd.," entered into a commercial contract with a supplier, "EcoGoods Corp.," for the procurement of industrial machinery. The contract was valued at ₹4 million and included terms related to the delivery of machinery, payment schedules, and performance guarantees.
'''

# Find the top 3 similar sections
similar_sections = find_similar_sections(query, faiss_index, df, top_k=3)

# Display the results
print(similar_sections[['chapter', 'Section']])
