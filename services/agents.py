from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import constants
import os

CHAT_COMPLETION_MODEL = constants.GPT_MODEL
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

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
    