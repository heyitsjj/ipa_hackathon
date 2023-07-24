from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import constants

# set up llm 
llm = OpenAI(temperature=0, openai_api_key=constants.OPENAI_API_KEY)

def generateSmartQuery(humanQuery:str):
    #Chain 1: Take in user questions 
    template = """{question}\n\n"""
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)

    #Chain 2: 
    template = """You are a financial chat assistant that specialized in helping Financial Advisors 
                    to analyze that funds and clients that they are responsible for. 
                    The Financial Advisor will ask questions about their funds, clients, and how these 
                    are affected by certain articles or measurement of funds' past performance. 
                    And you will be provided with the excerpt delimited by +++ below. 
                    The excerpt is a high-level overview of the structure and content of the documents you will later 
                    be provided with to answer further questions. 
                    You must use only the excerpt to answer the question. 

                    Here is the excerpt: 
                    +++
                    The insight documents are used my Financial Advisors to get insights on the 
                        financial/business opportunities and risks. 
                        This document contains 4 sections: \n
                        1) List of Global Investment Committee Investment Themes\n
                        2) Changes to Ideas for Investment Themes \n
                        3) Changes to Ideas for Asset Classes \n
                        4) Research Recommentations\n
                    
                    The report document is a comma-delimited csv document that contains the information 
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
                    +++

                    Here is a statement: 
                    {statement}

                    Make a bullet point list of the assumptions you made when producing the above statement. 
                    """
    prompt_template = PromptTemplate(input_variables=["statement"], template=template)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
    assumptions_chain_seq = SimpleSequentialChain(
        chains=[question_chain, assumptions_chain], verbose=True
    )

    answer = assumptions_chain_seq.run(humanQuery)

    return answer 




# Chain 3: Fact checking the assuptions 
    # template = """Here is a bullet point list of assertions:
    # {assertions}
    # For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    # prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    # fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
    # fact_checker_chain_seq = SimpleSequentialChain(
    #     chains=[question_chain, assumptions_chain, fact_checker_chain], verbose=True
    # )

    # # Final Chain: Generating the final answer to the user's question based on the facts and assumptions
    # template = """In light of the above facts, do you think this question '{}' is within the scope of the documents we have (as specified in the excerpt)?""".format(
    #     humanQuery
    # )
    # template = """{facts}\n""" + template
    # prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    # answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    # overall_chain = SimpleSequentialChain(
    #     chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    #     verbose=True,
    # )
