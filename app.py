from flask import Flask 
from api.chatbot import chatbot_api
from api.holdingSummaryTab import holdingSummary_api
import os, constants

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

app.register_blueprint(holdingSummary_api, url_prefix='/holdingSummary')
app.register_blueprint(chatbot_api, url_prefix='/chatbot')



