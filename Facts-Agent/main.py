from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 200,
    chunk_overlap=0
)


loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
    )

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

results = db.similarity_search("What is an interesting fact about ostrich?")

for result in results:
    print("\n")
    print(result.page_content)