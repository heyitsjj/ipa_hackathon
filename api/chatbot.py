from flask import Blueprint, request
from services.openai import callChatCompletionAPI
from flask import jsonify
from services.llama import getStorageContext
from llama_index import load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex, QuestionAnswerPrompt
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from flask import Flask
from services.smartquery import generateSmartQuery
from services.agents import multiCsvAgent
import pandas as pd
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain import PromptTemplate, LLMChain
import os


chatbot_api = Blueprint('chatbot', __name__)

@chatbot_api.route('/llamaQuery/FA_Report')
def llama_csv_query(): 
    indexPersistDir = "storage"

    storage_context = getStorageContext(indexPersistDir)
    loaded_index = load_index_from_storage(storage_context, index_id="report_index")

    query_engine = loaded_index.as_query_engine()

    queryReport = """Only output the column value of account_number where quantity column is less than 3"""
    response = query_engine.query(queryReport)
    
    responseStr = str(response)
    return jsonify({"content": responseStr})

@chatbot_api.route('/langchainQuery/FA_Summary')
def langchain_csv_summary(): 
    # Get the absolute path to the instance folder
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # Specify the path to your CSV file. 
    data_file = os.path.join(project_dir, '../data/hackathon.csv')
    # read csv
    df = pd.read_csv(data_file)

    # instantiate model, agent, and query 
    model = OpenAI(model_name="text-davinci-003", temperature=0.0)
    agent = create_pandas_dataframe_agent(model, df, agent="chat-zero-shot-react-description", verbose=True)
    # execute response 1
    responseStr1 = agent.run("What is the total dollar value of holdings for each individual client? Use tool python_repl_ast")
    # execute response 2
    responseStr2 = agent.run("What is the total Quantity and Value for each Fund? Use tool python_repl_ast")
    # execute response 3
    responseStr3 = agent.run("Please list the clients invested in each individual mutual fund. Use tool python_repl_ast")
    # use llm to summarize information into a paragraph
    templateResponse = responseStr1 + " " + responseStr2 + " " + responseStr3
    template = """Rewrite all of the following information into a traditional paragraph format. 
                Additionally, reorder names to follow the format firstname, lastname instead of lastname, firstname: 
                {templateResponse}""" 
    prompt = PromptTemplate(template=template, input_variables=["templateResponse"])
    llmRunner = LLMChain(prompt=prompt, llm=model)
    finalResponse = llmRunner.run(templateResponse)
    return jsonify({"content": finalResponse})

@chatbot_api.route('/llamaQuery/insight_and_report')
def llama_hack_query(): 
    # load index from storage 
    indexPersistDir = "storage"
    storage_context = getStorageContext(indexPersistDir)
    reportIndex = load_index_from_storage(storage_context, index_id="report_index")
    insightIndex = load_index_from_storage(storage_context, index_id="article_index")

    # build prompt 
    context_str = """The insight query engine contains a document that is used my Financial Advisors to get insights on the 
                        financial/business opportunities and risks. 
                        This document contains 4 sections: \n
                        1) List of Global Investment Committee Investment Themes\n
                        2) Changes to Ideas for Investment Themes \n
                        3) Changes to Ideas for Asset Classes \n
                        4) Research Recommentations\n
                    
                        The report query engine contains a comma-delimited csv document that contains the information 
                        about the clients and the funds that a specific Financial Advisor is 
                        in charge of. In other words, it contains the Financial Advisor's book of 
                        business in terms of what funds he is in charge of. 
                        The csv document contains 9 columns: 
                        1) FA_ID: this column contains the identification number of the Financial Advisor \n
                        2) FA_NAME: this column contains the Financial Advisor's name \n
                        3) ACCOUNT: this column contains the account number of the client \n
                        4) CLIENT_NAME: this column contains the Client's name \n
                        5) CUSIP: this column contains the identification number of a specific fund that is owned by the client and is managed by the Financial Advisor \n
                        6) FUND_NAME: this column contains the name fo the fund \n
                        7) TICKER: this column contains the ticker information of the fund that the Financial Advisor is in charge of \n
                        8) AMT_QTY: this column contains the quantity of a specific fund that the client owns and that the Financial Advisor is managing \n
                        9) AMT_MKVL: this column contains the market value of the amount of the specific funds that the client owns \n

                        Terminology - The below phrases are some common terms used by the Financial Advisors:
                        1) Book of Business: it means the funds that the financial advisor is responsible for. It can be found through the fund name, the cusip, and the ticker.  
                        2) Holding: it means the client name, quantity, market value, fund name, and ticker of the fund that the financial advisor is in charge of. 
                    """

    # question = """Give me only the research recomendations that are related to my book of business / the funds 
    #                 that the Financial Advisor is in charge for. In addition, also show me the holding information 
    #                 of the funds that are affected by the research recommendations. 
    #                 """
    
    table_json_format = """JSON. The JSON should follow the following structure and give the following data:
                        "data":[
                                {
                                    "FA_Name":"",
                                    "Client_Name":"",
                                    "Client_Account_Number":"",
                                    "Fund_Name":"",
                                    "Quantity":""
                                },
                            ]
                        
                        If the instruction is not clear, please see the following JSON delimited by === for an example output: 
                        ===
                        "data":[
                                {
                                    "FA_Name":"SMALL BARRY R."
                                    "Client_Name":"SKATTUM PATRICIA",
                                    "Client_Account_Number":"2015-09-23-15.36.26.059851",
                                    "Fund_Name":"Impax Global Environmental Markets Fund",
                                    "Quantity":"119.00"
                                },
                            ]
                        ===
                        Use the CLIENT_NME column for Client_Name. Use the ACCOUNT column for Client_Account_number. Use FUND_NAME column for Fund_Name. Use AMT_QTY column for Quantity. 
                        Do not combine clients even if they have the same fund. 
                        """
    
    question = f"""Give me the current holding information of the each fund that are affected the research recommendations in the format of {table_json_format}"""
    
    # question = "Give me only the Investment Ideas that are related to the funds / book of business that the financial advisor SMALL BARRY R. is in charge of."

    # question = "What are the research Recommendations? "

    # if "holding" in question: 
    #     question= f"{question}\n{table_json_format}"
    # else: 
    #     question = f"{question}text."
    

    query_str = f""" Use the following excerpt delimited by +++ as a context reference of the documents used: 
                    +++
                    {context_str}
                    +++
                    Answer the following question: 
                    {question}
                """

    # build query engine 
    insightEngine = insightIndex.as_query_engine()
    reportEngine = reportIndex.as_query_engine()

    # query tool for querying multiple docs 
    query_engine_tools = [
        QueryEngineTool(
            query_engine = insightEngine,
            metadata = ToolMetadata(name='insight', description='Provides information about insights on the business and financial opportunities for the Financial Advisors.')
        ),
        QueryEngineTool(
            query_engine=reportEngine,
            metadata=ToolMetadata(name='report', description='A csv file that contains information about the funds and client that a specific Financial Advisor have.')
        )
    ]

    # run query 
    s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)
    response = s_engine.query(query_str)

    responseStr = str(response)
    return jsonify({"content": responseStr})

#not working yet 
@chatbot_api.route('/testLangchain')
def test_langchain(): 
    question = "What are the research recommendations?"

    answer = str(generateSmartQuery(question))

    return jsonify({'langchain-content': answer})


@chatbot_api.route('/testMorningstarRating')
def test_agent():
    csv_file_1 = "data/hackathon.csv"
    csv_file_2 = "data/morningstar.csv"
    question = "List out the funds that are affected by the morningstar rating performance."
    # question = "List out the funds that are affected by the morningstar rating performance on 6/30/2023. "

    agent = multiCsvAgent(csv_file_1, csv_file_2)

    dict = agent.run(question)

    return dict

