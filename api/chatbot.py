from flask import Flask, Blueprint, request, render_template, url_for, flash, redirect
from codeDump_doNotUse.openai import callChatCompletionAPI
from flask import jsonify
from llama_index import load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex, QuestionAnswerPrompt
from langchain.agents import load_tools, initialize_agent, Tool
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from flask import Flask, request
from services.agents import *
import constants
from tools.qaAgentTools import *
from tools.responseFormater import formatResponse

chatbot_api = Blueprint('chatbot', __name__)

@chatbot_api.route('/chat', methods = ['POST'])
def chatFinal():
    # define llm 
    llm = ChatOpenAI(model=constants.GPT_MODEL,temperature=0)

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

    response = formatResponse(result)

    return response


@chatbot_api.route('/chatHere')
def chat():
    """ Test Book + Morningstar agent """
    result = formatResponse()

    return result

@chatbot_api.route('/prompt/', methods=('GET', 'POST'))
def prompt():
    if request.method == 'POST':
        role = request.form['role']
        query = request.form['query']

        # define llm 
        llm = ChatOpenAI(model=constants.GPT_MODEL,temperature=0)

        if not query:
            flash('Query prompt is required!')
        else:
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

        result = chat_chain.run(query)

        response = formatResponse(result)

        print(response)
        return render_template('prompt.html',result=response)

    return render_template('prompt.html')
