
import os
import hashlib

from Framework.utils.hash import my_hash
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

import datetime
from dotenv import load_dotenv
load_dotenv()

# Retrieve the Pinecone API key from user data
pinecone_key = os.environ.get('PINECONE_API_KEY')

# Initialize the Pinecone client with the retrieved API key
pc = Pinecone(
    api_key=pinecone_key
)

# Define constants for the Pinecone index, namespace, and engine
ENGINE = 'text-embedding-3-small'  # The embedding model to use (vector size 1536)
INDEX_NAME = 'semantic-search-rag'  # The name of the Pinecone index
NAMESPACE = 'default'  # The namespace to use within the index

# Initialize the OpenAI client with the API key from user data
client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)

# Function to get embeddings for a list of texts using the OpenAI API
def get_embeddings(texts:list, engine=ENGINE):
    # Create embeddings for the input texts using the specified engine
    response = client.embeddings.create(
        input=texts,
        model=engine
    )

    # Extract and return the list of embeddings from the response
    return [d.embedding for d in list(response.data)]

# Function to get embedding for a single text using the OpenAI API
def get_embedding(text, engine=ENGINE):
    # Use the get_embeddings function to get the embedding for a single text
    return get_embeddings([text], engine)[0]


def prepare_for_pinecone(texts, engine=ENGINE, document=None):
    # Get the current UTC date and time
    now = str(datetime.datetime.now(datetime.timezone.utc))

    # Generate vector embeddings for each string in the input list, using the specified engine
    embeddings = get_embeddings(texts, engine=engine)

    # Create tuples of (hash, embedding, metadata) for each input string and its corresponding vector embedding
    # The my_hash() function is used to generate a unique hash for each string, and the datetime.utcnow() function is used to generate the current UTC date and time
    responses = [
        (
            my_hash(text),  # A unique ID for each string, generated using the my_hash() function
            embedding,  # The vector embedding of the string
            dict(
                text=text, 
                date_uploaded=now,
                document=document
                )  # A dictionary of metadata
        )
        for text, embedding in zip(texts, embeddings)  # Iterate over each input string and its corresponding vector embedding
    ]

    return responses


def upload_texts_to_pinecone(texts, index, namespace=NAMESPACE, batch_size=None, show_progress_bar=False, document=None):
    # Call the prepare_for_pinecone function to prepare the input texts for indexing
    total_upserted = 0
    if not batch_size:
        batch_size = len(texts)

    _range = range(0, len(texts), batch_size)
    for i in _range if show_progress_bar else _range:
        text_batch = texts[i: i + batch_size]
        if document:
            prepared_texts = prepare_for_pinecone(text_batch, document=document)
        else:
            prepared_texts = prepare_for_pinecone(text_batch)


        # Use the upsert() method of the index object to upload the prepared texts to Pinecone
        total_upserted += index.upsert(
            vectors=prepared_texts,
            namespace=namespace
        )['upserted_count']


    return total_upserted


def query_from_pinecone(query, index, top_k=3, include_metadata=True):
    query_embedding = get_embedding(query, engine=ENGINE)

    return index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=include_metadata 
    ).get('matches')


def delete_texts_from_pinecone(texts, index, namespace=NAMESPACE):
    # Compute the hash (id) for each text
    hashes = [hashlib.md5(text.encode()).hexdigest() for text in texts]

    # The ids parameter is used to specify the list of IDs (hashes) to delete
    return index.delete(ids=hashes, namespace=namespace)


def delete_all(index, namespace=NAMESPACE):
    # TO delete all vectors, be careful!
    index.delete(delete_all=True, namespace=namespace) 
    return


def create_index(pinecone_key:str = None):
    import os
    from dotenv import load_dotenv
    load_dotenv()    
    
    # Retrieve the Pinecone API key from user data
    if pinecone_key==None or pinecone_key=='':
        pinecone_key = os.environ.get('PINECONE_API_KEY')

    # Initialize the Pinecone client with the retrieved API key
    pc = Pinecone(
        api_key=pinecone_key
    )
    
    if INDEX_NAME not in pc.list_indexes().names():  # need to create the index
        pc.create_index(
            name=INDEX_NAME,  # The name of the index
            # namespace=NAMESPACE,
            dimension=1536,  # The dimensionality of the vectors for our OpenAI embedder
            metric='cosine',  # The similarity metric to use when searching the index
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    return pc.Index(name=INDEX_NAME)
    