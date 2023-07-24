from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext
)
from llama_index.llms import OpenAI
import openai
import constants
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.node_parser import SimpleNodeParser

openai.api_key = constants.OPENAI_API_KEY


#Load documents
def loadDocuments(directoryPath:str): 
    documents = SimpleDirectoryReader(directoryPath, filename_as_id=True).load_data()
    return documents


# Create and Store Index 
def createAndStoreIndex(documents): 
    # create storage context using default stores 
    storage_context = StorageContext.from_defaults(
        docstore = SimpleDocumentStore(),
        vector_store = SimpleVectorStore(),
        index_store = SimpleIndexStore(),
    )

    # create parser and parse document into nodes 
    parser = SimpleNodeParser()
    faReport = []
    article = []
    for doc in documents: 
        if doc.id_.split(".")[1].startswith("csv"):
            faReport.append(doc)
        
        if doc.id_.split(".")[1].startswith("txt"):
            article.append(doc)

    articleNodes = parser.get_nodes_from_documents(article)
    reportNodes = parser.get_nodes_from_documents(faReport)

    # build index (insights and report)
    articleIndex = VectorStoreIndex(articleNodes, storage_context=storage_context)
    reportIndex = VectorStoreIndex(reportNodes, storage_context=storage_context)

    #set index id 
    articleIndex.set_index_id("article_index")
    reportIndex.set_index_id("report_index")

    # save index 
    articleIndex.storage_context.persist(persist_dir="storage")
    reportIndex.storage_context.persist(persist_dir="storage")



# Get Storage Context for index loading 
def getStorageContext(persist_dir:str):
    storage_context=StorageContext.from_defaults(
        persist_dir=persist_dir
    )
    return storage_context





