import os
from dotenv import load_dotenv

from crewai_tools import tool
import yfinance as yf

from crewai import Task

from crewai import Agent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

from crewai import Crew
from crewai.process import Process
from langchain_openai import ChatOpenAI

import streamlit as st

from pydantic import BaseModel
from typing import List

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

# modifiy default model of crewAI
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

#====================Output format=====================
class RoeIndicator(BaseModel):
    ROE_value : float
    ROE_report : str

class PerIndicator(BaseModel):
    PER_value : float
    PER_report : str

class FfoIndicator(BaseModel):
    FFO_trending : str
    FFO_report : str

class IndicatorReport(BaseModel):
    ROE : RoeIndicator
    PER : PerIndicator
    FFO : FfoIndicator

class DividendPayoutRatioReport(BaseModel):
    Dividend_Payout_Ratio : float
    Dividend_Payout_Ratio_report : str

class StockNews(BaseModel):
    title : str
    summary : str
    sentiment : str

class StockPriceReport(BaseModel):
    stock_price_trending : str
    stock_price_report : str

class AnalysisReport(BaseModel):
    company: str
    decision : str
    indicator_report : IndicatorReport
    dividend_payout_ratio_report : DividendPayoutRatioReport
    company_new_report : List[StockNews]
    stock_price_report : StockPriceReport
    conclusion : str

#====================Agents=====================
class Tools:
    @tool("Get ROE(Return on Equity) rate of company")
    def get_ROE(symbol):
        """
        Useful to get ROE(Return on Equity) of company.
        Use this when you have to know ROE(Return on Equity) rate of company.
        The input of this tool should be a symbol of company,for example AAPL, NET, TSLA etc...
        """

        stock = yf.Ticker(symbol)
        value = str(stock.info['returnOnEquity'] * 100) + "%"

        return f"ROE Rate of {symbol} : {value}"
    
    @tool("Get PER rate of company")
    def get_PER(symbol):
        """
        Useful to get PER(Price Earning Ratio) of company.
        Use this when you have to know PER(Price Earning Ratio) rate of company.
        The input of this tool should be a symbol of company,for example AAPL, NET, TSLA etc...
        """

        stock = yf.Ticker(symbol)
        value = stock.info['trailingPE']

        return f"PER Rate of {symbol} : {value}"
    
    @tool("Get FFO rate of company")
    def get_FFO(symbol):
        """
        Useful to get FFO value of company.
        Use this when you have to know FFO(Funds From Operations) rate of company.
        The input of this tool should be a symbol of company,for example AAPL, NET, TSLA etc...
        """

        stock = yf.Ticker(symbol)

        nonOperatingIncome = list(stock.income_stmt.loc[['Other Non Operating Income Expenses']].dropna(axis=1).to_dict().values())
        keys = stock.income_stmt.loc[['Net Income']].dropna().columns
        netIncome = list(stock.income_stmt.loc[['Net Income']].dropna(axis=1).to_dict().values())

        date = list()
        for k in keys:
            date.append(str(k).replace(" 00:00:00", ""))

        FFO = dict()

        for net, non, d in zip(netIncome, nonOperatingIncome, date):
            FFO[d] = net['Net Income'] - non['Other Non Operating Income Expenses']
    
        return f"FFO of {symbol} : {FFO}"

    @tool("Get Dividend payout ratio of company")
    def get_diviend_payout_ratio(symbol):
        
        """
        A tool to get Dividend Payout Ratio of company.
        The input of this tool should be a symbol of company,for example AAPL, NET, TSLA etc...
        """

        stock = yf.Ticker(symbol)
        value = str(round(stock.info['payoutRatio'] * 100, 2)) + "%"

        return f"Dividend Payout Ratio of {symbol} : {value}"
    
    @tool("Get News of Company")
    def get_news(symbol):
        """
        Useful to get URLs of news articles related to a stock.
        The input of this tool should be a symbol of company,for example AAPL, NET, TSLA etc...
        """

        stock = yf.Ticker(symbol)
        return list(map(lambda x: x["link"], stock.news))
    
    @tool("Get Stock Price")
    def get_stock_price(symbol):
        """
        A tool to get 5 years stock price history as csv file format.
        The input of this tool should be a symbol of company,for example AAPL, NET, TSLA etc...
        """

        stock = yf.Ticker(symbol)

        price_history = stock.history(period="5y")[["High"]].to_csv()   

        return price_history

class Tasks:
    
    def symbol_research(self, agent):
        return Task(
            description="""Gather stock symbol of {company}.""",
            expected_output="Your Final Answer MUST be a stock symbol. You MUST NOT find other information about company.",
            agent=agent,
        )
    def indicator_research(self, agent):
        return Task(
            description="Gather information of three company indicator. You should look ROE, PER, FFO rate of {company}. Your job is to order other agents. You should carefully look and give works to appropriate workers.",
            expected_output=f"Your Final Answer must be summary of three company indicator. Ask each three company indicator to each three analyst. You should summarize each of analyst's report.",
            agent=agent,
        )    
    def dividend_payout_ratio_analysis(self, agent):
        return Task(
            description="Get value of dividend payout ratio  of {company}. And judge whether {company} is good to invest",
            expected_output=f"Your Final Answer Must be company's dividend rate and judgement whether this company is good to invest.",
            agent=agent,
        )   
    def news_analysis(self, agent):
        return Task(
            description="Gather and analyze the latest news and market sentiment surrounding the stock of {company}. Provide a summary of the news and any notable shifts in market sentiment",
            expected_output=f"Your final answer MUST be a detailed summary of important news and market sentiment surrounding the stock. Include any notable shifts in market sentiment and provide insights on how these factors could impact the stock's performance",
            agent=agent,
        )  
    def stock_price_analyst(self, agent):
        return Task(
            description="Conduct a detailed technical analysis of the price movements of {company}'s stock and trends identify key support and resistance levels, chart patterns, and other technical indicators that could influence the stock's future performance. Use historical price data and technical analysis tools to provide insights on potential entry points and price targets.",
            expected_output=f"Your final answer MUST be a detailed technical analysis report that includes key support and resistance levels, chart patterns, and technical indicators. Provide insights on potential entry points, price targets, and any other relevant information that could help your customer make informed investment decisions.",
            agent=agent,
        )  
    def hedge_fund_manager(self, agent, context):
        return Task(
            description="Based on the response of other agents, provide a detailed investment recommendation for {company}'s stock. Include your analysis of the stock's potential risks and rewards, and provide a clear rationale for your recommendation",
            expected_output=f"Your final answer MUST be a detailed investment recommendation report to BUY or SELL the stock that includes your analysis of the stock's potential risks and rewards. Provide a clear rationale for your recommendation based on the research, technical analysis, and financial analysis reports",
            agent=agent,
            context=context,
            output_pydantic=AnalysisReport,
        )
    def translation(self, agent, context):
        return Task(
            description="Base on given context, translate the context with Korean.",
            expected_output=f"Your final answer MUST be a Korean Report that contain information on given context.",
            agent=agent,
            context=context,
        )

class Agents:
    def symbol_researcher(self):
        return Agent(
            role="Symbol_Researcher",
            goal="Search and find company's symbol. ",
            backstory="""You're skilled in finding company's stock symbol. You have to find company's stock symbol with company's name. If you cannot find information, just say 'I don't know'. Do not MAKEUP your owns
                        Here are some examples. 
                        Human : What is Apple stock symbol?
                        Your answer : AAPL

                        Human : What is Intel stock symbol?
                        Your answer : INTC

                        Human : What is {company} stock symbol?
                        Your answer :
                    """,
            tools = [
                SerperDevTool(),
                ScrapeWebsiteTool()
            ],
            allow_delegation = False,
            verbose=True,
        )
    
    def company_indicator_analyst(self):
        return Agent(
            role="Company Indicator Analyst",
            goal="Your task is to act as a 'Coordinator Agent.' You will not solve the problem directly but delegate tasks to specialized agents. These agents will handle different parts of the problem.",
            backstory="""You are talented person to gather the best insights by relying on specialized agents. You should carefully choose agents and give appropriate work to each agents. If you think it is hard to accomplish task, just say I don't know.""",
            allow_delegation = True,
            verbose=True
        )
    
    def ROE_analyst(self):
        return Agent(
            role="ROE(Return on Equity) analyst",
            goal="Analyses ROE(Return on Equity) rate of given company. You should get exact value of Return On Equity rate relying on given tools.",
            backstory="""
                    You are good at summarizing some information. You should get exact value of Return on Equity rate relying on given tools. 
                    Here are some context you can use. Use this context to analysing ROE rate.
                    Context : ROE measures the efficiency and profitability of a company's equity investment. A higher ROE indicates more effective management and profitability, with a benchmark of 15% being desirable. Companies with high ROE are often well-managed and show potential for long-term growth.
                    
                    Finally, you should give recommandation based on your ROE(Return On Equity) analysis. 

                    ==Caution==
                    Do Not MAKEUP.
                    Do Not Calculate your own. Use Tools
                    You MUST only use given context and tools.
                    """,
            tools = [
                Tools.get_ROE,
            ],
            allow_delegation = False,
            verbose=True,

            
        )
    
    def PER_analyst(self):
        return Agent(
            role="PER(Price Earning Ratio) analyst",
            goal="Analyses PER(Price Earning Rate) rate of given company. You should get exact value of Price Earning Ratio rate relying on given tools.",
            backstory="""
                    You are good at summarizing some information. You should get exact value of Price Earning Ratio rate relying on given tools. 
                    Here are some context you can use. Use this context to analysing PER rate.
                    Context : PER assesses if a stock is undervalued or overvalued by comparing the stock price to its EPS. A PER below 1 suggests the stock may be undervalued, while a PER above 1 indicates possible overvaluation. Investing in stocks with a PER below 1 is generally seen as a wise choice.
                    
                    Finally, you should give recommandation based on your PER(Price Earning Ratio) analysis. 

                    ==Caution==
                    Do Not MAKEUP.
                    Do Not Calculate your own. Use Tools
                    You MUST only use given context and tools.
                    """,
            tools = [
                Tools.get_PER,
            ],
            allow_delegation = False,
            verbose=True
        )
    
    def FFO_analyst(self):
        return Agent(
            role="FFO(Funds From Operations) analyst",
            goal="Analyses FFO(Funds From Operations) rate of given company. You should get exact value of Funds From Operations rate relying on given tools.",
            backstory="""
                    You are good at summarizing some information. You should get exact value of Fund From Operations rate relying on given tools. 
                    Here are some context you can use. Use this context to analysing FFO rate.
                    Context : FFO reflects earnings from core operations by adjusting net income for non-operational gains and expenses. It's a more accurate measure of a company’s operational performance compared to net income. Companies with consistently growing FFO over the years are considered good investment opportunities.

                    Finally, you should give recommandation based on your FFO(Funds From Operations) analysis. 

                    ==Caution==
                    Do Not MAKEUP.
                    Do Not Calculate your own. Use Tools
                    You MUST only use given context and tools.
                    """,
            tools = [
                Tools.get_FFO,
            ],
            allow_delegation = False,
            verbose=True
        )
  
    def dividend_payout_ratio_analyst(self):
        return Agent(
            role="Dividend Payout Ratio Analyst",
            goal="Analyses the dividend payout ratio of company and provides insight of this company. If you cannot find information, just say 'I don't know'. Do not MAKEUP your owns",
            backstory="You are an expert in judging dividend payout ratio. Get dividend payout ratio of company and check out dividend payout ratio is higher than 40%. If dividend payout ratio is higher than 40%, this company is good to invest.",
            
            tools = [
                Tools.get_diviend_payout_ratio,
            ],
            allow_delegation = False,
            verbose=True
        )
    
    def news_analyst(self):
        return Agent(
            role="News Analyst",
            goal="Analyses the news of given company. Get the sentiment of each news and analysis if this company is good to invest. You MUST get important articles of {company}.",
            backstory="""
                        ==Role==
                        You are an News Analyst. 
                        You analysis sentiment with each news. 

                        ==Task==
                        Summarize that important news and give insight whether it is good to invest this company depending on News.
                        *If there is no Link for news, just say none.*

                        ==Caution==
                        You are giving a advice to important customer.
                        If you cannot find information, just say 'I don't know'. 
                        Do not MAKEUP your owns.
                        """,
            
            tools = [
                Tools.get_news,
                SerperDevTool(),
                ScrapeWebsiteTool(),

            ],
            allow_delegation = False,
            verbose=True,
        )
    
    def stock_price_analyst(self):
        return Agent(
            role="Stock Price Analyst",
            goal="Analyses past and recent stock history.",
            backstory="""
                ==Role==
                You are an expert in Stock Price Analysis. 
                
                ==Task==
                You should analysis stock price and give response to customer if this company is good to invest.
                You should judge company stock price with following three statement. 
                
                Statement
                - First, company's stock price increasement should be higher than market inflation. 
                - Second, stock price should not be decrease 15% in recent 1 years. 
                - All of this statement is satisfy, this company is good to invest. 

                ==Caution==
                You are giving a advice to important customer.
                If you cannot find information, just say 'I don't know'. 
                Do not MAKEUP your owns.
                
                """,
            
            tools = [

            ],
            allow_delegation = False,
            verbose=True,
        )
    
    def hedge_fund_manager(self):
        return Agent(
            role="Hedge Fund Manager",
            goal="""
                ==Task==
                Manages a portfolio of stocks and makes strategic investment decisions to maximize returns.
                Use insights from Traditional Indicator Analyst, Dividend Payout Ratio Analyst, News Analyst, Stock Price Analyst. 

                ==Cautions==
                Also, You MUST CONSIDER importance of each data.    
            """,
            backstory="""
                    ==Role==
                    You are the head fund manager. 
                    
                    ==Task==
                    You need to decide whether to buy this company's stock by listening to the opinions of the Traditional Indicator Analyst, Dividend Rate Analyst, News Analyst, Market Share Analyst, and Stock Price Analyst. 
                    The importance of each piece of information varies. 
                    The importance decreases from left to right, meaning the priority of information is as follows
                    three indicator(ROE, PER, FFO) > dividend payout ratio > news > stock price. 
                    
                    ==Caution==
                    You are currently dealing with a very important client. You MUST be put your foot down.
                    Do Not ask question to other researcher. 
                    Do not MAKEUP your owns.
                """,
            
            allow_delegation = False,
            verbose=True,

        )
    
    def translater(self):
        return Agent(
            role="Translater",
            goal="""Translate Report to Korean.""",
            backstory="""
                    ==Role==
                    You are talented translater. 
                    You are specialized in translating to Korean.
                    
                    ==Task==
                    You will receive English Json Report. You should translate English report to Korean report.
                    You Must use honorifics with Korean.

                    ==Caution==
                    You are currently dealing with a very important client.
                    Do Not ask question to other researcher. 
                    Do not MAKEUP your owns.
                """,
            
            allow_delegation = False,
            verbose=True,
        )

agents = Agents()
task = Tasks()

symbolResearcher = agents.symbol_researcher()
companyIndicatorAnalyst = agents.company_indicator_analyst()
dividendRateAnalyst = agents.dividend_payout_ratio_analyst()
ROE_analyst = agents.ROE_analyst()
PER_analyst = agents.PER_analyst()
FFO_analyst = agents.FFO_analyst()
newsAnalyst = agents.news_analyst()
stockPriceAnalyst = agents.stock_price_analyst()
hedgeFundManager = agents.hedge_fund_manager()
#translater = agents.translater()

symbolResearchTask = task.symbol_research(symbolResearcher)
indicatorResearchTask = task.indicator_research(companyIndicatorAnalyst)
dividendPayoutRatioAnalysisTask = task.dividend_payout_ratio_analysis(dividendRateAnalyst)
newsAnalysisTask = task.news_analysis(newsAnalyst)
stockPriceAnalystTask = task.stock_price_analyst(stockPriceAnalyst)
hedgeFundManagerTask = task.hedge_fund_manager(hedgeFundManager, [indicatorResearchTask, dividendPayoutRatioAnalysisTask, newsAnalysisTask])
#translationTask = task.translation(translater, [hedgeFundManagerTask])

list_agents = [symbolResearcher, companyIndicatorAnalyst, ROE_analyst, PER_analyst, FFO_analyst, dividendRateAnalyst, newsAnalyst, stockPriceAnalyst, hedgeFundManager]
list_tasks = [symbolResearchTask, indicatorResearchTask, dividendPayoutRatioAnalysisTask, newsAnalysisTask, stockPriceAnalystTask, hedgeFundManagerTask]

crew = Crew(
    agents=list_agents,
    tasks=list_tasks,
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4o-mini"),
    memory=True
)

@st.cache_resource(show_spinner="Gather Information...")
def kick(company):
    result = crew.kickoff(inputs=dict(company=f"{company}"))

    return result

def new_formattor(result):
    new_text = list()
    for index, new in enumerate(result['company_new_report']):
        new_text.append(
            f"""**{index + 1}. {new['title']}** <br>
            - Summary : {new['summary']} <br>
            - Sentiment : {new['sentiment']} <br>""".replace("$", ""))
    return '<br>'.join(new_text)

st.set_page_config(
    page_icon="📊",
    page_title="Stock Analysis",
)

st.markdown(
    """
    # Stock Analysis

    Write down the name of a company and our Agent will do the research for you. 

    **Be Patient with the result!** 
    """

)

stock = st.text_input("Write the name of the global company you are interested on.")

st.divider()

if stock:
    result_pydantic = kick(stock)

    result = result_pydantic.dict()

    st.markdown(
        f"""
        ## Stock Recommendation for {result['company']}

        ### Recommendation : {result['decision']}

        ### Indicator
        - **ROE(Return on Equity)**
            - {result['indicator_report']['ROE']['ROE_value']}%
            - {result['indicator_report']['ROE']['ROE_report']}
        
        - **PER(Price to Earning Ratio)**
            - {result['indicator_report']['PER']['PER_value']}
            - {result['indicator_report']['PER']['PER_report']}

        - **FFO(Fund from operations)**
            - {result['indicator_report']["FFO"]['FFO_trending'].replace("$", "")}
            - {result['indicator_report']['FFO']['FFO_report'].replace("$", "")}

        ### Dividend Payout Ratio
        - Payout Ratio : **{result['dividend_payout_ratio_report']['Dividend_Payout_Ratio']}%**
        - {result['dividend_payout_ratio_report']['Dividend_Payout_Ratio_report'].replace("$", "")}

        ### Company News
        {new_formattor(result)}

        ### Stock Price
        - Stock Price Trending : {result['stock_price_report']['stock_price_trending'].replace("$", "")}
        - {result['stock_price_report']['stock_price_report'].replace("$", "")}

        ### Conclusion
        - {result['conclusion'].replace("$", "")}


        """, unsafe_allow_html=True)
