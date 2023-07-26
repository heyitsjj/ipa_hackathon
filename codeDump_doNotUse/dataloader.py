from flask import Blueprint, request
from services.openai import callChatCompletionAPI
from flask import jsonify
from codeDump_doNotUse.llama import loadDocuments, createAndStoreIndex, getStorageContext
from llama_index import load_index_from_storage
import os

dataloader_api = Blueprint('dataloader', __name__)

@dataloader_api.route('/')
def load_data():
    #load all data and store index when FA enter the root of the UI 
    # dataDir = "data"
    dataDir = "data"

    documents = loadDocuments(dataDir)
    id = createAndStoreIndex(documents)

    indexIsCreated = os.path.exists("storage")

    return jsonify({"content": indexIsCreated})