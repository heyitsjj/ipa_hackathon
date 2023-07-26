from flask import Blueprint
import pandas as pd 
import json, constants

holdingSummary_api = Blueprint('holdingSummary', __name__)

@holdingSummary_api.route('/')
def getHoldingSummary():
    # Read csv and take only info related to specific FA 
    df_bob = pd.read_csv(constants.path_bob)

    # get total assets 
    bob_total = df_bob[df_bob["FAName"] == constants.FA_Name]["MarketValue"].sum()

    # get advisory assets 
    df_bob_adv = df_bob[(df_bob["FAName"]==constants.FA_Name) & (df_bob["TradingPlatform"]=="ADVISORY")]
    bob_adv = df_bob_adv["MarketValue"].sum()

    # get non-advisory assets 
    df_bob_non_adv = df_bob[(df_bob["FAName"]==constants.FA_Name) & (df_bob["TradingPlatform"]=="NON-ADVISORY")]
    bob_non_adv = df_bob_non_adv["MarketValue"].sum()

    # generate json response 
    holdingSummaryResponse = json.loads(constants.holdingSummaryJson)
    holdingSummaryResponse["response"]["data"]["TotalAssets"] = bob_total
    holdingSummaryResponse["response"]["data"]["Advisory"] = bob_adv
    holdingSummaryResponse["response"]["data"]["NonAdvisory"] = bob_non_adv

    return holdingSummaryResponse