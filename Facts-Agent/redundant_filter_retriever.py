from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

# This is custom retriever that will control the duplication of the documents that we add in our vector store.
class RedundantFilterRetriever(BaseRetriever):
    # we are defining embeddings and chroma as fields up here so that anyone who is initialising this class would pass these values
    # The idea is to make this class generic so that it will be used for any embeddings and chroma objest that has defined persist_directory
    embeddings: Embeddings 
    chroma: Chroma

    def get_relevant_documents(self, query):
        #calculate embeddings for query string
        emb = self.embeddings.embed_query(query)
    
        # take embeddings and feed that to max_marginal_relevence_search_by_vector

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    
    async def aget_relevant_documents(self):
        return []