from flask import Flask 
from api.chatbot import chatbot_api
# from codeDump_doNotUse.dataloader import dataloader_api
import os, constants

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# app.register_blueprint(dataloader_api)
app.register_blueprint(chatbot_api, url_prefix='/chatbot')



