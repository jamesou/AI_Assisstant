import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
# import cohere
from langchain.llms import Cohere
 

# Load environment variables from .env file
load_dotenv()

# Set the Hugging Face API key from the environment variable
os.environ['huggingfaceid'] = os.getenv('hf_token')
cohere_token = os.getenv('cohere_token')
anthropic_token= os.getenv('anthropic_token')

#Set up the LLM for text generation
def setup_llm():
    # model_name = "mistralai/Pixtral-12B-2409"  # Token Limit: 8192 tokens
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # text_generation_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=150,
    #     device=0 if torch.cuda.is_available() else -1
    # )
    # llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    # llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=8192, temperature=0.7, token=os.environ['huggingfaceid'])

    llm = Cohere(model="command", cohere_api_key=cohere_token,max_tokens=2048, temperature=0.7)
    return llm

#Set up RetrievalQA chain
def setup_rag(llm,query):
    # user_query = "Nestech"
    # Load a pre-trained model for embedding generation
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Generate the embedding for the query
    # query_embedding = embedding_model.embed_query(user_query)

    client = QdrantClient("http://localhost:6333")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_collection",
        embedding=embedding_model
    )
    
    # Search for documents similar to the query
    search_results = vector_store.similarity_search(query=query)

    #Output the retrieved results and their scores
    # for doc in search_results:
    #     print(f"* {doc.page_content} [{doc.metadata}]")
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return rag


# Function to generate the answer using a prompt template
def answer_question(retrieval, prompt_template,llm, query):
    print(f"query:{query}")
    # Retrieve the relevant documents
    response = retrieval.invoke({"query": query})
    source_docs = response['source_documents']
    # print(f"source_docs:{source_docs}")
    # Combine the context (retrieved documents)
    context = "\n\n".join([doc.page_content for doc in source_docs])
    print(f"context:{context}")
    llm_chain = prompt_template | llm
    # Now, pass this final prompt to the LLMChain for generation
    generated_answer = llm_chain.invoke({"context": context, "question": query})
    
    # Display the final answer
    print(f"generated_answer:\n{generated_answer}\n")

# Set up prompt template
template = """
You are a helpful assistant that answers questions based on the provided context.
If you can't find the answer from context, you can answer it based on your pre-trained model.

Context: {context}
Question: {question}

"""
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

 # Set up LLM
llm = setup_llm()

# Interactive Q&A
while True:
    query = input("Enter your question (or type 'exit' to quit): ")
    if query.lower() in ['exit', 'quit']:
        break
    # Set up RAG
    rag = setup_rag(llm,query=query)
    answer_question(rag,prompt_template,llm, query)