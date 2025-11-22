# Explanation for RAG (Retrieval-Augmented Generation) Module Integration

This document outlines the steps to integrate the RAG (Retrieval-Augmented Generation) module into the existing document management system. The RAG module enhances the system's capabilities by allowing users to ask questions about their uploaded documents and receive contextually relevant answers.

## 1) Chunking module :

The chunking module is responsible for breaking down large documents into smaller, manageable pieces called chunks. This is essential for efficient processing and retrieval of information. The chunking process typically involves:

Function for chunking senteces : 
```python
def simple_chunk_by_sentences(text: str, sentences_per_chunk: int = 5, min_chunk_length: int = 50) -> List[str]:
```

- check if text is empty or only whitespaces return empty list
- Define sentence enders using regex pattern to avoid splitting within abbreviations or numbers.
- Split the text into sentences using the defined sentence enders using `sentences = re.split(sentence_endings, text)`
- Clean the sentences by stripping leading and trailing whitespaces and filtering out empty sentences using `cleaned_sentences = [s.strip() for s in sentences if s.strip()]`
- Initialize `chunks = []` to store the resulting chunks.
- Enter a for loop i.e: `for i in range(0, len(sentences), sentences_per_chunk):` and create chunks by `chunk_sentences = sentences[i:i + sentences_per_chunk]`
- in each iteration join the selected sentences into a single chunk using `chunk = " ".join(chunk_sentences)`
- Check if the chunk length is greater than or equal to min_chunk_length before appending to chunks list.
- Return the list of chunks.

## 2) Embeddings module:

The embeddings module converts text chunks into vector representations that can be efficiently searched and compared. This is crucial for the retrieval aspect of RAG. The embedding process typically involves:

- SentenceTransformer from sentence_transformers library to create embeddings.
- The model used is 'all-MiniLM-L6-v2' for generating dense vector representations of text chunks.

Function 
```python
def generate_embedding(texts: List[str]) -> List[List[float]]:
```

- Initialize the SentenceTransformer model using `model = SentenceTransformer('all-MiniLM-L6-v2')`
- Generate embeddings for the provided chunks using `embeddings = model.encode(chunks, convert_to_numpy=True)`
- `return embeddings.tolist()`

Function for batch embedding generation : 
```python
def generate_embeddings(texts: List[str], batch_size: int = 16) -> List[List[float]]:
```

- Initialize an empty list `all_embeddings = []` to store embeddings.
- Enter a for loop to process texts in batches using `for i in range(0, len(texts), batch_size):`
- Extract the current batch using `batch = texts[i:i + batch_size]`
- Generate embeddings for the current batch using `batch_embeddings = self.generate_embedding(batch)`
- Extend the all_embeddings list with the batch embeddings using `all_embeddings.extend(batch_embeddings)`
- Return the complete list of embeddings.

## 4) vector_store module:

The vector_store module manages the storage and retrieval of vector representations of text chunks. It allows for efficient searching of relevant chunks based on user queries. The vector store process typically involves:

- Using FAISS (Facebook AI Similarity Search) for efficient similarity search and clustering of dense vectors.
- Connect to Qdrant vector database for storing and retrieving vector embeddings.

Function to create a collection in Qdrant : 
```python
create_collection_if_not_exists(collection_name: str):
```

- Check if the collection already exists using `existing_collections = self.client.get_collections().collections`
- If the collection does not exist, create it using `self.client.create_collection(collection_name, vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE))`
- Create index using document ID as primary key.

Function to store document chunks : 
```python
def store_document_chunks(self, document_id: int, chunks: List[str], embeddings: List[List[float]],collection_name:str = None) -> int:
```

- Check if collection_anme is None, set it to default using settings and also call create_collection_if_not_exists
- prep the embeddings and chunks for uploading 
- Enter a for loop : `for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):` to loop through each chunk and its corresponding embedding.
- Create a point using 
```python
point = PointStruct(
id=str(uuid4()),
vector=embedding,
payload={
"document_id": document_id,
"chunk_index": i,
"text": chunk,
"chunk_length": len(chunk)
}
)
```
- Upload the points to Qdrant using `client.upsert_points(collection_name, points)`
- return the number of points 

Functon to search relevant chunks : 
```python
def search_similar_chunks(query:str, top_k:int =5, collection_name:str = None,document_id:int = None) -> List[Dict[]]:
```

- First check for collection name and assign default if None
- Generate embedding for the query using generate_embedding
- Create a filter if document_id is provided using `filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])` else `filter = None`
- Perform the search using `search_result = self.client.search_points(collection_name=collection_name, query_vector=query_embedding, limit=top_k, filter=filter)`
- Return the search results as a list of dictionaries containing chunk text and metadata.
- i.e 
```python
return [
{
"text": result.payload["text"],
"document_id": result.payload["document_id"],
"chunk_index": result.payload["chunk_index"],
"chunk_length": result.payload["chunk_length"],
"score": result.score
}
for result in search_result
]
```

## 5) QA module:

The QA (Question Answering) module leverages the RAG architecture to provide answers to user queries based on the retrieved document chunks. The QA process typically involves:

- Using a pre-trained language model such as Gemini to generate answers based on the context provided by the retrieved chunks.
- The function `answer_query(query: str, document_id: int = None, top_k: int = 5) -> dict:` is responsible for generating answers to user queries.
- Search for relevant chunks using vector_store's search_similar_chunks method if no relevant chunks are found return a message indicating no relevant information is available.
- Build context by joining the context chunks into a single string and enumarate them for reference. eg "[Excerpt 1] : text ...".
- Write a prompt for the language model that includes the user query and the constructed context.
- Use the language model to generate an answer based on the prompt,`response = model.generate_content(prompt)`
- Return the generated answer along with the context used for reference.