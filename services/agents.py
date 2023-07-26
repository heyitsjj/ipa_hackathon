from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_pandas_dataframe_agent,initialize_agent, LLMSingleActionAgent, AgentOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import constants, os, json
import pandas as pd
from services.openai import callCompletionAPI
    
# insight txt QA chain 
def insight_qa_chain(question:str):
    # Specify llm 
    llm = OpenAI(model=constants.COMPLETION_MODEL,temperature=0.0)

    # load chain 
    insight_qa_chain = load_qa_chain(llm, chain_type="refine", verbose=True, return_intermediate_steps=True)

    # load insight txt 
    with open(constants.path_insight) as f: 
        insight_txt=f.read()

    # split text into chunk to avoid token issue 
    text_splitter = CharacterTextSplitter(
        chunk_size=constants.text_chunk_size, 
        chunk_overlap=constants.text_overlap, 
        length_function=len
    )
    splitted_insight_doc = text_splitter.create_documents([insight_txt])

    # specify question and prompt 
    prompt = f"""
                {constants.systemInstruction}\n

                Here is the excerpt: 
                +++
                {constants.insightContext}\n
                +++

                Answer the following question in a very brief and concise way: 
                {question}
                """

    response = insight_qa_chain(
        {"input_documents": splitted_insight_doc, "question": prompt},
        return_only_outputs=True
    )

    return response["output_text"]



# book of business csv dataframe agent 
def bob_df_agent(question:str):
    # Specify llm 
    llm = OpenAI(model=constants.COMPLETION_MODEL,temperature=0.0)

    # Get the absolute path to the instance folder 
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify path to the CSV file 
    bookOfBusinessFile = os.path.join(project_dir,'../data/bookOfBusiness.csv')

    # Read csv and take only info related to specific FA 
    df = pd.read_csv(bookOfBusinessFile)
    df = df[df['FAName'].str.contains(constants.FA_Name)]

    # create agent 
    bobAgent = create_pandas_dataframe_agent(
        llm=llm, 
        df=df, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        max_execution_time=constants.max_execution_time,
        max_iterations=constants.max_iterations
    )

    # specify prompt 
    prompt = f"""
                {constants.systemInstruction}\n

                Here is the excerpt: 
                +++
                {constants.bookOfBussinessContext}\n
                {constants.terminologyContext}\n
                +++

                Answer the following question: 
                {question}\n

                Important Note: You must strictly and only return the answer in the below json array format directly. Do not improvise. And stop thinking / observing once you think you know the final answer.: 
                {constants.tableFormat}
                """
    #Important Note: You should strictly and only return the answer in the below json array format directly: {constants.tableFormat}

    response = bobAgent.run(prompt)

    return response


# book of business + morningstar csv dataframe agent 
def bob_ms_df_agent(question:str):
    # Specify llm 
    llm = OpenAI(model=constants.COMPLETION_MODEL,temperature=0.0)

    # read csv and get morningstar dataframe 
    df_ms = pd.read_csv(constants.path_morningstar)

    # get splitted / chunked book of business dataframe 
    df_bob_list = get_chunk_bob_dataframe()

    # specify prompt and question 
    # question = "What are the holding information in my book of business where the fund has a 5 star morningstar rating? "
    prompt = f"""
                {constants.systemInstruction}\n

                Here is the excerpt: 
                +++
                {constants.bookOfBussinessContext}\n
                {constants.terminologyContext}\n
                {constants.morningstarContext}\n
                +++

                Here is the conditions: 
                If the user did not ask for the holding information in the question, then you final answer must only contain the Fund names and the tickers that corresponds to the funds.\n

                Answer the following question: 
                {question}
                """
    
    #Check if it is asking for holding information
    
    # loop through book of business and call df agent 
    result = ""
    for bob in df_bob_list: 
        bob_ms_agent = create_pandas_dataframe_agent(
            llm,
            [df_ms,bob],
            verbose=True,
        )
        response = bob_ms_agent.run(prompt)
        result = f"{result}\n{response}"

    # json_convert_prompt = """
    #                         Your job is to convert the following text string (delimited by ===) into one of the below JSON arrays depending on whether the text string contains only 
    #                         Fund name and ticker or the text string contains more information (including Account number, client name, fund name, etc.) than that.

    #                         Here are the two possible json formats:
    #                         1) {"response":{"type":"table","data":[{"FundName":"","Ticker":""}]}} \n 
    #                         2) {"response":{"type":"table","data":[{"AccountNumber":"","ClientName":"","FundName":"","Ticker":"","Quantity":"","MarketValue":""}]}} \n
    #                         """ + f"""
    #                         Here is the text string that you need to parse into one of the above json format: 
    #                         ===
    #                         {result}
    #                         ===

    #                         You must only return the final json without any other additional sentence or words. 
    #                         """
    # response = callCompletionAPI(json_convert_prompt)
    # # final_json = response['choices'][0]['text']

    return result


# Morningstar csv dataframe agent 
def get_morningstar_df_agent():
    # Specify llm 
    llm = OpenAI(model=constants.COMPLETION_MODEL,temperature=0.0)

    # Read csv 
    df = pd.read_csv(constants.path_morningstar)

    # create agent 
    morningstarAgent = create_pandas_dataframe_agent(
        llm=llm, 
        df=df, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        max_execution_time=constants.max_execution_time,
        max_iterations=constants.max_iterations)

    return morningstarAgent


# to do: add summary content for bob (Done: endpoint /holdingSummary)

# to do: add cross context with insight + book of business (Jingjing)

# to do: add cross context with book of business + focus list (Jingjing)

# to do: handle general finance questions (Jingjing)

# to do: handle non-sense / non-finance questions (Jingjing)


# utility: used to chunk large book of business dataframe 
def get_chunk_bob_dataframe():
    # Read csv and take only info related to specific FA 
    df_bob = pd.read_csv(constants.path_bob)
    df_bob = df_bob[df_bob['FAName'].str.contains(constants.FA_Name)]

    # split / chunk book of business dataframe 
    chunk_size = constants.df_chunk_size
    df_bob_list = [df_bob[i:i+chunk_size] for i in range (0,df_bob.shape[0],chunk_size)]

    return df_bob_list


# def get_json_from_response_chain(response:str):
#     # define llm 
#     llm = OpenAI(model=constants.COMPLETION_MODEL,temperature=0)

#     previous_response = response

#     instruction = """
#                 Your job is to convert the following text string (delimited by ===) into one of the below JSON arrays depending on whether the text string contains only 
#                 Fund name and ticker or the text string contains more information (including Account number, client name, fund name, etc.) than that.

#                 Here are the two possible json formats:
#                 1) {"response":{"type":"table","data":[{"FundName":"","Ticker":""}]}} \n 
#                 2) {"response":{"type":"table","data":[{"AccountNumber":"","ClientName":"","FundName":"","Ticker":"","Quantity":"","MarketValue":""}]}} \n
#                 """
    
#     template = instruction + f""" Here is the text string that you need to parse into one of the above json format: 
#                                 ===
#                                 {previous_response}
#                                 ==="""
#     # template = f""" 
#     #         Your job is to convert the following text string (delimited by ===) into one of the below JSON arrays depending on whether the text string contains only 
#     #         Fund name and ticker or the text string contains more information (including Account number, client name, fund name, etc.) than that.

#     #         Here are the two possible json formats:
#     #         1) {"response":{"type":"table","data":[{"FundName":"","Ticker":""}]}} \n 
#     #         2) {"response":{"type":"table","data":[{"AccountNumber":"","ClientName":"","FundName":"","Ticker":"","Quantity":"","MarketValue":""}]}} \n
            
#     #         Here is the text string that you need to parse into one of the above json format: 
#     #         ===
#     #         {previous_response}
#     #         ===
#     #     """
    
#     # specify prompt for converting response to Json 
#     # prompt = PromptTemplate(
#     #     input_variables=["previous_response"],
#     #     template=template
#     # )

#     prompt = PromptTemplate(template)

#     json_converter_chain = LLMChain(llm=llm, prompt=prompt)

#     final_json = json_converter_chain(return_only_outputs=True)

#     return final_json


# def test_single_action (response:str):









# def get_json_from_response_agent(response:str):
#     # define llm 
#     llm = OpenAI(model=constants.COMPLETION_MODEL,temperature=0)

#     # get all tools 
#     tools = [
#             ]

#     # initiate agent 
#     json_converter = initialize_agent(
#         llm=llm,
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=True
#     )

#     # Specify prompt 
#     instruction =""" 
#             Your job is to convert the following text string (delimited by ===) into one of the JSON arrays depending on whether the text string contains only 
#             Fund name and ticker or the text string contains more information (including Account number, client name, fund name, etc.) than that.

#             Here are the two possible json format: 
#             1) {"response":{"type":"table","data":[{"AccountNumber":"","ClientName":"","FundName":"","Ticker":"","Quantity":"","MarketValue":""}]}} \n
#             2) {"response":{"type":"table","data":[{"FundName":"","Ticker":""}]}} \n
#             """
    
#     prompt = f"""{instruction} 
#             Here is the text string that you need to parse into one of the above json formats: 
#             ===
#             {response}
#             === """
    
#     final_json = json_converter.run(prompt)

#     return final_json



# Here are the two possible json format: 
#             1) {
#                     "response":{
#                         "type":"table",
#                         "data":[
#                             {
#                                 "AccountNumber":"",
#                                 "ClientName":"",
#                                 "FundName":"",
#                                 "Ticker":"",
#                                 "Quantity":"",
#                                 "MarketValue":""
#                             }
#                         ]
#                     }
#                 }
#             2) {
#                     "response":{
#                         "type":"table",
#                         "data":[
#                             {
#                                 "FundName":"",
#                                 "Ticker":""
#                             }
#                         ]
#                     }
#                 }






    


    