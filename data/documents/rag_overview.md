# Introduction to Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that improves the
factual accuracy and relevance of language model outputs by grounding them in
retrieved evidence from an external knowledge base. Instead of relying solely
on parametric knowledge encoded in a model's weights during training, RAG
dynamically fetches relevant documents at inference time and provides them to
the language model as context alongside the user's query.

The term was introduced in a 2020 paper by Lewis et al. from Facebook AI Research,
titled "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."
The paper demonstrated that combining a retrieval component with a seq2seq model
could outperform models that relied purely on parametric memory for open-domain
question answering tasks.

## Why RAG Reduces Hallucination

Large language models (LLMs) hallucinate because their parametric knowledge has
a fixed cutoff date and their internal representations are lossy compressions of
training data. When asked a factual question outside their training distribution,
they produce plausible-sounding but incorrect answers.

RAG addresses this in two ways:

1. **Grounded context**: The retrieved passages contain the actual information
   needed. The model is instructed to answer using only that context, not prior
   knowledge.

2. **Traceable citations**: Each retrieved chunk is associated with a source
   document and page number. The generated answer can cite [1], [2] etc.,
   making it auditable.

## The Retrieval Pipeline

A typical RAG retrieval pipeline works as follows:

1. **Ingestion phase** (offline, done once or periodically):
   - Load source documents (PDFs, web pages, text files).
   - Split each document into overlapping chunks of approximately 200-500 tokens.
   - Embed each chunk using a dense encoder (e.g., all-MiniLM-L6-v2) to produce
     a fixed-dimensional vector (384 dimensions for all-MiniLM-L6-v2).
   - Store (chunk text, vector, metadata) in a vector database such as ChromaDB,
     Pinecone, Weaviate, or FAISS.

2. **Query phase** (online, per user query):
   - Embed the user's question with the same encoder used during ingestion.
   - Perform an approximate nearest-neighbour (ANN) search in the vector store
     to retrieve the top-K most semantically similar chunks.
   - Concatenate the retrieved chunks as context and pass them with the query
     to the generative LLM (e.g., Gemini, GPT-4, Claude).
   - The LLM produces an answer grounded in the retrieved context.

## ChromaDB as a Vector Store

ChromaDB is an open-source, embeddable vector database designed for RAG
applications. Key features include:

- **Persistent storage**: ChromaDB can persist its HNSW index and metadata to
  disk using a `PersistentClient`, enabling the indexed collection to survive
  process restarts without re-embedding.

- **Cosine similarity**: Collections can be configured with `hnsw:space=cosine`
  to use cosine similarity rather than Euclidean distance, which is more
  appropriate for normalised sentence embeddings.

- **Explicit embeddings**: When supplying pre-computed embeddings during
  `upsert()`, ChromaDB stores and indexes them directly without invoking an
  internal embedding function. This keeps the embedding model consistently
  controlled by application code.

- **Metadata filtering**: Each stored record can carry an arbitrary metadata
  dictionary. During retrieval, `where` clauses can filter by source filename,
  page number, date, or any other stored field.

## The all-MiniLM-L6-v2 Embedding Model

The `all-MiniLM-L6-v2` model from the sentence-transformers library is a
popular lightweight choice for RAG applications because:

- It produces 384-dimensional dense vectors.
- Its maximum context length is 256 WordPiece tokens.
- It can run on CPU without GPU hardware, making it suitable for local
  development and resource-constrained environments.
- The model was fine-tuned on over 1 billion sentence pairs using a contrastive
  learning objective, giving it strong semantic similarity performance despite
  its compact size.

A critical consequence of the 256-token limit is that text chunks fed to this
model must not exceed approximately 220-254 usable tokens (accounting for the
special [CLS] and [SEP] tokens added automatically). Exceeding this limit causes
silent truncation of the tail of the chunk, which degrades retrieval quality for
longer passages.

## Chunking Strategy

Text chunking converts raw document text into embedding-ready segments.
Two key parameters determine the chunking behaviour:

- **chunk_size**: The target number of tokens per chunk. Should be chosen to
  stay within the embedding model's input limit with a safety margin.
- **overlap**: The number of tokens shared between adjacent chunks. Overlap
  ensures that sentences at chunk boundaries are captured by at least one chunk,
  preventing retrieval gaps at boundaries.

A sliding window with overlap=45 tokens on a 220-token chunk means that
each step advances 175 tokens, and the final 45 tokens of chunk N are the
first 45 tokens of chunk N+1. This preserves contextual continuity.

## Evaluation

A RAG system can be evaluated on two dimensions:

1. **Retrieval quality**: Precision@K — of the top-K retrieved chunks, how many
   are actually relevant to the query? Measured manually or with a labelled
   ground-truth dataset.

2. **Generation quality**: Does the answer faithfully summarise the retrieved
   context? Does it avoid introducing facts not present in the retrieved chunks?
   This can be assessed by comparing the RAG answer to a plain LLM answer on
   the same question and checking for hallucinations.

## Summary

RAG is now the dominant architecture for building production AI assistants that
must be accurate, up-to-date, and auditable. It combines the fluency of
generative LLMs with the factual grounding of information retrieval, reducing
hallucination without the cost of fine-tuning. The key components are a dense
embedding model, a vector store, and a capable generative model \u2014 all three of
which are available in lightweight, locally-runnable forms suitable for a college
assignment.
