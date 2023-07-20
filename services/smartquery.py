from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import constants

# set up llm 
llm = OpenAI(temperatire=0, openai_api_key=constants.OPENAI_API_KEY)

# def generateSmartQuery(humanQuery:str):
#     #Chain 1: Take in user questions 
#     template = """{question}\n\n"""
#     prompt_template = PromptTemplate(input_variables=["question"], template=template)
#     question_chain = LLMChain(llm=llm, prompt=prompt_template)

#     #Chain 2: 
#     template = f"""You are a financial chat assistant that specialized. 
#                     The """