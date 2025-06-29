from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever = retriever,
    chain_type="stuff"
)

result = chain.run("What is an interesting fact about ostrich?")

print(result)