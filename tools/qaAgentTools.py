from langchain.tools import BaseTool
from pydantic import BaseModel, Field 
import constants
from services.agents import *
from typing import Type
from tools.responseFormater import formatResponse

class HumanInput(BaseModel):
    question: str = Field(description="The input question string")

#insight tool 
class InsightSummary (BaseTool): 
    name = "Insight Summary Tool"
    description = constants.tool_insight_description 
    args_schema: Type[BaseModel] = HumanInput

    def _run(self,question:str) -> str: 
        result = insight_qa_chain(question)
        return result
    
    def _arun(self, question: str):
        raise NotImplementedError("This tool does not support async")
    

#bob tool 
class BookOfBusinessAnalysis(BaseTool):
    name = "Book of Business Analysis Tool"
    description = constants.tool_bob_description
    args_schema: Type[BaseModel] = HumanInput

    def _run(self,question:str) -> str:
        result = bob_df_agent(question)
        return result 
    
    def _arun(self, question: str):
        raise NotImplementedError("This tool does not support async")
    
# bob and morningstar tool 
class MorningstarAndBookOfBusinessAnalysis(BaseTool):
    name = "Morningstar rating and Book of Business Analysis"
    description = constants.tool_bob_ms_description 
    args_schema: Type[BaseModel] = HumanInput

    def _run(self,question:str) -> str: 
        result = bob_ms_df_agent(question)
        return result
    
    def _arun(self, question: str):
        raise NotImplementedError("This tool does not support async")

    
