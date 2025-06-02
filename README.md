# Retrieval Augmented Generation (RAG)

All rights Reserved \
2025 [Israel Llorens](https://www.linkedin.com/in/israel-llorens/)

## Configuration

In order to execute this project, you need to place an .env file with the following:

```bash
GREMLIN_SERVER_URL=ws://localhost:8182/gremlin
PINECONE_API_KEY=<PINECONE API KEY>
OPENAI_API_KEY=<OPEN AI API KEY>
...
# 
```

## Development Architecture

For this project's implementation, I leveraged Apache Spark as the core data processing engine to efficiently load and process large volumes of text files. Spark handles the preprocessing, text normalization, and chunking operations across distributed workers, then orchestrates the embedding generation through OpenAI's API with proper batching and rate limiting.

The resulting embeddings and metadata are systematically stored in Pinecone's vector database.

This architecture creates a robust foundation where Spark handles the computational heavy lifting for data preparation, while Pinecone provides fast, scalable vector storage and retrieval capabilities essential for real-time query processing in my healthcare and telecommunications applications.

## Project Purpose

This is a sample project designed as a proof of concept for a potential full-scale implementation. This project has as intention to provide knowledge on how to implement a simple RAG Sytem using various techniques. In order to build a Vector Database, we have implemented FAISS (locally) or Pinecone Free tier service.

Its main goal is to evaluate the skills in:

- Natural Language Processing (NLP)
- Information Retrieval (IR)
- Generative AI
- and/or Semantic Search
- Generate answers from stored values.

## Future RAG-Specific Improvements

This project represents my initial exploration into RAG systems - it's a working proof of concept, but there's significant room for enhancement to achieve production-level scalability and performance.

- *Adaptive chunking approaches*: I plan to expand beyond basic text splitting by incorporating semantic chunking that respects document structure, alongside configurable fixed-size and sentence-boundary strategies. This will allow me to optimize chunk sizes based on document type and use case.

- *Multi-model embedding support*: Currently tied to OpenAI's embedding models, I want to add flexibility to experiment with different embedding approaches - from open-source alternatives like BGE or E5 to domain-specific models that might better capture medical or financial terminology relevant to my work.

- *Retrieval quality metrics*: To properly evaluate and tune the system, I need to implement standard information retrieval metrics. NDCG will help assess ranking quality, MRR will measure how quickly relevant documents appear, and recall@k will ensure I'm not missing important context.

- *Response quality assessment*: Beyond just retrieving relevant documents, I want to measure whether the generated responses are actually grounded in the retrieved content (faithfulness) and whether they truly address the user's question (relevance). This is crucial for applications in healthcare and finance where accuracy is paramount.

- *Graph RAG implementation*: Instead of isolated vector embeddings, it is possible to implement Graph RAG using LangChain to construct knowledge graphs that capture relationships and hierarchical structures within documents. LangChain's graph tools would automatically extract entities and relationships, enabling multi-hop reasoning across connected concepts rather than just similarity matching. This approach would provide more contextually aware retrieval and better explainability by showing how information pieces connect to form answers.

These improvements will transform this from a demonstration into a robust system suitable for real-world applications across the domains I work in.

### This analysis aims to extract meaningful insights from the text file documents in order to help answer questions with information within the text documents library
