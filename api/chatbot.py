from flask import Blueprint, request
from services.openai import callChatCompletionAPI
from flask import jsonify
from services.llama import getStorageContext
from llama_index import load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex, QuestionAnswerPrompt
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from flask import Flask


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

    query_str = """Give me only the research recomendations that are related to my book of business / the funds 
                    that the Financial Advisor is in charge for. In addition, also show me the holding information 
                    of the funds that are affected by the research recommendations. 
                    """

    general_prompt_tmpl = (
        """We have provided context information about this document below \n
        -----------------------\n
        {context_str}
        -----------------------\n
        Do not use your own knowledge. Use only the above context and the documents provided, """
        f"please answer {query_str}"
    )

    general_prompt = QuestionAnswerPrompt(general_prompt_tmpl)

    # build query engine 
    insightEngine = insightIndex.as_query_engine(
        text_qa_template=general_prompt
    )
    reportEngine = reportIndex.as_query_engine(
        text_qa_template=general_prompt
    )

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

