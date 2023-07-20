from flask import Flask 
from api.chatbot import chatbot_api
from api.dataloader import dataloader_api

app = Flask(__name__)


app.register_blueprint(dataloader_api)
app.register_blueprint(chatbot_api, url_prefix='/chatbot')



