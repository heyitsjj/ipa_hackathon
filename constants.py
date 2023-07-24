# Financial Advisor ID/Name 
FA_Name = "FA-1"

#GPT Connectivity 
GPT_MODEL = "gpt-3.5-turbo"
COMPLETION_MODEL = "text-davinci-003"
OPENAI_API_KEY = "xxxxx"

chatCompletionURL = "https://api.openai.com/v1/chat/completions"
completionURL = "https://api.openai.com/v1/completions"
embeddingURL = "https://api.openai.com/v1/embeddings"

#execution / iteration limit 
max_execution_time=60
max_iterations=30

#dataframe chunking size 
df_chunk_size = 3
text_chunk_size = 1000
text_overlap = 100

# data file path and name 
path_bob = "data/bookOfBusiness.csv"
path_insight = "data/insights.txt"
path_morningstar = "data/morningstar.csv"


#Context Constant 
systemInstruction = """
                    You are a financial chat assistant that specialized in helping Financial Advisors 
                    to analyze the funds and clients that he is responsible for. 
                    The Financial Advisor will ask questions about their funds, clients, and how these 
                    are affected by certain articles or measurement of funds' past performance. 
                    And you will be provided with the excerpt delimited by +++ below. You should only use the 
                    excerpt as a context along with the document provided when answering the user's questions. 
                    """

bookOfBussinessContext = """
                        The Book of Bussiness document contains the information 
                        about the clients and the funds that a specific Financial Advisor is 
                        in charge of. \n
                        It contains 9 columns: \n
                        1) FAName: this column contains the name of the Financial Advisor \n
                        2) AccountNumber: this column contains the account number of the client that the Financial Advisor is responsible for \n
                        3) TradingPlatform: this column contains information about the trading platform of the fund \n
                        4) 459_ClientName: this column contains the name of the Client that the Financial Advisor is responsible for \n
                        5) Cusip: this column contains the identification number of the fund that is owned by the client and is managed by the Financial Advisor \n
                        6) FundName: this column contains the name fo the fund that is owned by the client and is managed by the Financial Advisor\n
                        7) Ticker: this column contains the ticker information of the fund that is owned by the client and is managed by the Financial Advisor\n
                        8) Quantity: this column contains the quantity of the specific fund that is owned by the client and is managed by the Financial Advisor \n
                        9) MarketValue: this column contains the market value of the fund that is owned by the client and is managed by the Financial Advisor 
                        """

insightContext = """
                The Insight and Research document contains information that is used my Financial Advisors to get insights on the financial/business opportunities and risks. 
                This document contains 4 sections: \n
                    1) List of Global Investment Committee Investment Themes\n
                    2) Changes to Ideas for Investment Themes \n
                    3) Changes to Ideas for Asset Classes \n
                    4) Research Recommentations
                 """

morningstarContext = """
                    See the exerpt delimited by === for a background knowledge about Morningstar Ratings: \n
                    ===
                    The Morningstar rating is a ranking given to publicly traded mutual funds and exchange traded funds by the investment research firm called Morningstar.
                    Funds receive ratings ranging from 1 to 5, with 1 given to the worst performers and 5 for the best. The ranking is based on variations in a fund's monthly returns. 
                    ===\n

                    See the excerpt delimited by --- for the context of the Morningstar rating data document that is provided to you: \n
                    ---
                    The Morningstar data document contains information of the morningstar rating of each fund on different days. It contains 5 columns: 
                    1) Date: this column contains the date of when the morningstar rating is given to the fund 
                    2) Ticker: this column contains the ticker information of the fund 
                    3) FundName: this column contains the name fo the fund 
                    4) MorningStarRating: this column contains the actual morningstar rating of the fund on a certain day 
                    5) RatingsChange: this column contains the rating change of the fund compared to its previous morningstar rating. The value of the column can be same, upgrade, or downgrade. 
                    --- 
                    """

terminologyContext = """
                    Terminology - The below phrases are some common terms used by the Financial Advisors:\n
                    1) Book of Business: it means the information of the clients and the funds that a Financial Advisor is responsible for.\n  
                    2) Holding: it means that the user wants you to provide the values of the following columns that satisfy the conditions of the user's question: AccountNumber, 459_ClientName, Cusip, FundName, Ticker, Quantity, and MarketValue. \n
                    3) Top N Holdings: it means that the user wants the top N holdings basing on how large their market values are. \n 
                    4) Orphan Funds / Position: it means any fund under one client that has a quantity value below 30. 
                    """

# tool descriptions 
# tool_insight_description = """Takes the user's question as input and summarize or/and retrieve a part of the financial insight document basing on the question. It returns text."""
# tool_bob_description = """Takes query to retrieve the holding information basing on the analysis of book of business and returns dataframe of holdings. """
# tool_bob_ms_description = """Takes query to retrieve the holding information basing on the morningstar rating from dataframe input and returns dataframe of holdings. """

tool_insight_description = """Useful when user wants to summarize or retrieve a part of the Financial Insight text document. """
tool_bob_description = """Useful when user wants to retrieve or analyze the holdings in his book of business. The analysis includes: top N holdings, orphan position, etc. """
tool_bob_ms_description = """Useful when user wants to retrieve the holding information basing on the morningstar rating from datafrom input."""


# response json format 
tableJsonInstruction = """ Please format the answer into the following JSON format: """
tableResponseFormat = """
                        {
                            "response":{
                                "type":"table",
                                "data":[
                                    {
                                        "AccountNumber":"",
                                        "ClientName":"",
                                        "Cusip":"",
                                        "FundName":"",
                                        "Ticker":"",
                                        "Quantity":"",
                                        "MarketValue":""
                                    }
                                ]
                            }
                        }
                        """

tableFormat = """
                {
                    "data":[
                        {
                            "AccountNumber":"",
                            "ClientName":"",
                            "Cusip":"",
                            "FundName":"",
                            "Ticker":"",
                            "Quantity":"",
                            "MarketValue":""
                        }
                    ]
                }
            """

textJsonInstruction = """ Please format the answer into the following JSON format: """
textResponseFormat = """
                    {
                            "response":{
                                "type":"text",
                                "data":{
                                    "textResponse":""
                                }
                            }
                        }
                    """