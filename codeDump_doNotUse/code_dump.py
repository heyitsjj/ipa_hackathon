def bob_ms_df_chain():
    # Specify llm 
    llm = OpenAI(model=constants.COMPLETION_MODEL,temperature=0.0)

    # Load QA Chain 
    qa_chain = load_qa_chain(llm=llm, chain_type="map_reduce", verbose=True, return_intermediate_steps=True)

    # Get the absolute path to the instance folder 
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # create csv array that contains morningstar and book of business  
    morningstarFile = os.path.join(project_dir,'../data/morningstar.csv')
    bobFile = os.path.join(project_dir,'../data/bookOfBusiness.csv')

    # read csv 
    df_ms = pd.read_csv(morningstarFile)
    df_bob = pd.read_csv(bobFile)
    df_bob = df_bob[df_bob['FAName'].str.contains(constants.FA_Name)]

    # split / chunk book of business dataframe 
    chunk_size = 3
    df_bob_list = [df_bob[i:i+chunk_size] for i in range (0,df_bob.shape[0],chunk_size)]

    question = "What are the holding information in my book of business where the fund has a 5 star morningstar rating? "
    prompt = f"""
                {constants.systemInstruction}\n

                Here is the excerpt: 
                +++
                {constants.bookOfBussinessContext}\n
                {constants.terminologyContext}\n
                {constants.morningstarContext}\n
                +++

                Answer the following question: 
                {question}
                """
    result = ""
    for bob in df_bob_list:
        result = qa_chain(
            {"input_documents": [df_ms, bob], "question": prompt},
            return_only_outputs=True
            )
    
    final = result["output_text"]

    return final 





# def aggregate():
#     #aggregate result 

# def chunk_df():










# ====================================================================
# def bob_ms_csv_agent():
#     # Specify llm 
#     llm = ChatOpenAI(model=constants.GPT_MODEL,temperature=0.0)

#     # Get the absolute path to the instance folder 
#     project_dir = os.path.dirname(os.path.abspath(__file__))

#     # create csv array that contains morningstar and book of business  
#     morningstarFile = os.path.join(project_dir,'../data/morningstar.csv')
#     bobFile = os.path.join(project_dir,'../data/bookOfBusiness.csv')
#     csvArr = [morningstarFile, bobFile]

#     # create multi csv agent 
#     bob_ms_agent = create_csv_agent(
#         ChatOpenAI(model=constants.GPT_MODEL,temperature=0.0),
#         csvArr,
#         verbose=True,
#         agent_type=AgentType.OPENAI_FUNCTIONS
#     )

#     return bob_ms_agent



# ====================================================================
# create a single-csv agent 
def singleCsvAgent(csvPath:str): 
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model=CHAT_COMPLETION_MODEL),
        csvPath,
        verbose=True, 
        agent_type=AgentType.OPENAI_FUNCTIONS
    )
    return agent

# create multi-csv agent 
def multiCsvAgent(csvPath1, csvPath2):
    csvFileArr = [csvPath1, csvPath2]

    agent = create_csv_agent(
        OpenAI(temperature=0),
        csvFileArr,
        verbose=True, 
    )

    return agent






# ====================================================================
# =============================Chatbot.py==============================
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

    # agent = multiCsvAgent(csv_file_1, csv_file_2)

    # dict = agent.run(question)

    # return dict


# question = "Which fund that I am in charge of has a 5 star morningstar rating? "
    # question = "Give me the Morningstar 5 star rated funds that are present in my book of business on 7/24/2023." 
    # question = "What are the holding information in my book of business where the fund has a 5 star morningstar rating? "
    # prompt = f"""
    #             {constants.systemInstruction}\n

    #             Here is the excerpt: 
    #             +++
    #             {constants.bookOfBussinessContext}\n
    #             {constants.terminologyContext}\n
    #             {constants.morningstarContext}\n
    #             +++

    #             Answer the following question: 
    #             {question}
    #             """
    
    # agent = bob_ms_df_agent()

    # dict = agent.run(prompt)

    # return dict



    """ Test Morningstar Agent """
    # question = "What are the funds that have a 5 star rating for Morningstar Rating? "
    # prompt = f"""
    #             {constants.systemInstruction}\n

    #             Here is the excerpt: 
    #             +++
    #             {constants.bookOfBussinessContext}\n
    #             {constants.terminologyContext}\n
    #             +++

    #             Answer the following question: 
    #             {question}
    #             """
    
    # agent = morningstar_df_agent()

    # dict = agent.run(prompt)

    # return dict


    """ Test Book of Business Agent """
    # # question = "Show me the holding information of all orphan positions in Financial Advisor FA-1's book of business."
    # question = "Give me the clients who has an orphan position. "
    # # question = "Give me the names of all clients. Use tool python_repl_ast"
    # prompt = f"""
    #             {constants.systemInstruction}\n

    #             Here is the excerpt: 
    #             +++
    #             {constants.bookOfBussinessContext}\n
    #             {constants.terminologyContext}\n
    #             +++

    #             Answer the following question: 
    #             {question}
    #             """
    
    # agent = bob_df_agent()

    # dict = agent.run(prompt)

    # return dict


     #=== insight only 
    # question = "Summarize the Global Investment Committee Themes." 

    #=== insight + business tool: test business 
    # question = "Give me the clients who has an orphan position. "
    # question = "Give me the top 3 holdings."

    #=== insight + business + bob_ms tool: 
    # test insight 
    # question = "Summarize the Global Investment Committee Themes." 
    #test business 
    # question = "Give me the top 3 holdings."
    # question = "Give me the clients who has an orphan position. "
    # test bob_ms 
    # question = "Which funds that I am in charge of has a 5 star morningstar rating?"
    # question = "Give me the ticker of the Morningstar 5 star rated fund/funds that are present in my book of business on 7/24/2023."