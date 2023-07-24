from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, create_csv_agent, load_tools, create_pandas_dataframe_agent
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
import constants, os 
import pandas as pd

CHAT_COMPLETION_MODEL = constants.GPT_MODEL

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

                Answer the following question: 
                {question}
                """
    
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


# to do: add cross context with insight + book of business (Jingjing)

# to do: add cross context with book of business + focus list (Jingjing)

# to do: handle general finance questions (Jingjing)

# to do: handle non-sense / non-finance questions (Jingjing)


# utility: used to chunk large book of business dataframe 
def get_chunk_bob_dataframe():
    # Get the absolute path to the instance folder 
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Read csv and take only info related to specific FA 
    df_bob = pd.read_csv(constants.path_bob)
    df_bob = df_bob[df_bob['FAName'].str.contains(constants.FA_Name)]

    # split / chunk book of business dataframe 
    chunk_size = constants.df_chunk_size
    df_bob_list = [df_bob[i:i+chunk_size] for i in range (0,df_bob.shape[0],chunk_size)]

    return df_bob_list





    


    