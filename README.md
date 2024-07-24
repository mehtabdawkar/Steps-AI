# RAPTOR Indexing and Retrieval System
**Overview**
This project implements a RAPTOR indexing system that processes text from PDF files, chunks the text, creates embeddings using SBERT, clusters the embeddings using Gaussian Mixture Models (GMM), and stores the summarized clusters in a Milvus database. Additionally, it provides enhanced retrieval techniques using query expansion, hybrid retrieval methods (BM25 and BERT/DPR), re-ranking using GPT-3.5-turbo, and question-answering capabilities.


**Dependencies**
Ensure you have the following dependencies installed:
pip install PyPDF2 nltk torch transformers scikit-learn openai rank_bm25 pymilvus

**File Structure**
ML PROJECT.ipynb: Contains the main code for RAPTOR indexing, retrieval, and question answering.
Genesis.pdf, For-the-Win.pdf, Crime-and-Punishment-.pdf: Sample PDF files to be processed.

**Title and Links for the selected Textbooks for content extraction:**
Title: Genesis.pdf
Link: https://manybooks.net/titles/genesis-0

Title: For-the-Win.pdf
Links: https://manybooks.net/titles/doctorowother10for_the_win.html

Title : Crime-and-Punishment-.pdf
Links: https://manybooks.net/titles/dostoyevetext018crmp10.html

**Usage**
**1. Configuration**
Set up your OpenAI API key in the **ML PROJECT.ipynb** file:
openai_api_key = 'your-openai-api-key'
client = OpenAI(api_key=openai_api_key)

**2. Connecting to Milvus**
Ensure Milvus is running and set up the connection in the **ML PROJECT.ipynb** file:
connections.connect("default", host="localhost", port="19530")

**3. Processing PDFs**
The following function processes each PDF, extracts text, chunks it, creates embeddings, and performs RAPTOR indexing:
def process_textbook(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    raptor_index(chunks, embeddings)
To process multiple PDFs, use:
pdf_paths = ['Genesis.pdf', 'For-the-Win.pdf', 'Crime-and-Punishment-.pdf']

for pdf_path in pdf_paths:
    process_textbook(pdf_path)


**4. Querying the RAPTOR Index**
To perform a query, retrieve and re-rank the data, and generate answers:
query = "What is the impact of social media on youth?"
chunks = chunk_text(extract_text_from_pdf('Genesis.pdf'))
embeddings = embed_chunks(chunks)
retrieved_chunks = hybrid_retrieval(query, chunks, embeddings)
ranked_chunks = rerank_retrieved_data(query, retrieved_chunks)
answer = answer_question(query, ranked_chunks)

print("Top relevant chunks:")
for chunk in ranked_chunks:
    print(chunk)

print("\nAnswer to the question:")
print(answer)


**Functions**
**Text Processing**
extract_text_from_pdf(pdf_path): Extracts text from a given PDF file.
chunk_text(text, max_tokens=100): Chunks the text into 100-token chunks while preserving sentence boundaries.

**Embedding and Clustering**
embed_chunks(chunks): Embeds the text chunks using SBERT.
cluster_embeddings(embeddings, n_clusters=5): Clusters the embeddings using Gaussian Mixture Models.

**Summarization and Indexing**
summarize_clusters(chunks, soft_labels, n_clusters): Summarizes the clustered chunks using GPT-3.5-turbo.
raptor_index(chunks, embeddings, depth=3, current_depth=0): Recursively indexes the text chunks and their embeddings.

**Retrieval and Question Answering**
expand_query(query): Expands the query using synonym expansion.
hybrid_retrieval(query, chunks, embeddings): Performs hybrid retrieval combining BM25 and DPR.
rerank_retrieved_data(query, retrieved_chunks): Re-ranks the retrieved data using GPT-3.5-turbo.
answer_question(query, ranked_chunks): Generates answers to the query using the re-ranked data and GPT-3.5-turbo.
