import json, constants

responseJsonFormat = """
                        {
                            "response":{
                                "type":"",
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

def formatResponse(result:str):
    try: 
        # table view response 
        finalResponse = json.loads(result)
    except ValueError as e: 
        # text response 
        finalResponse = json.loads(textResponseFormat)
        finalResponse["response"]["data"]["textResponse"] = result

    return finalResponse