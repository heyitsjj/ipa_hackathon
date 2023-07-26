from flask import Blueprint, request
from services.openai import callChatCompletionAPI
from flask import jsonify
from llama_index import load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex, QuestionAnswerPrompt
from langchain.agents import load_tools, initialize_agent, Tool
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from services.agents import *
import constants
from tools.qaAgentTools import *
from tools.responseFormater import formatResponse

chatbot_api = Blueprint('chatbot', __name__)

@chatbot_api.route('/chat', methods = ['POST'])
def chatFinal():
    # define llm 
    llm = OpenAI(model=constants.COMPLETION_MODEL,temperature=0)

    # get user question
    requestJson = request.get_json()
    question = requestJson["userQuestion"]
    
    # get all tools 
    tools = [InsightSummary(),
             BookOfBusinessAnalysis(),
             MorningstarAndBookOfBusinessAnalysis()
            ]

    # initiate agent 
    chat_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    result = chat_chain.run(question)

    # response = formatResponse(result)

    return result





# test / experiment endpoint 
# @chatbot_api.route('/chatHere')
# def chat():
#     result = holding_summary()

#     return result
