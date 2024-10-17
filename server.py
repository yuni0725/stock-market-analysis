import os
from dotenv import load_dotenv
import asyncio

import yfinance as yf

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.output_parsers import PydanticOutputParser

import streamlit as st

import json

from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# modifiy default model of crewAI
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

llm = ChatOpenAI(
    temperature = 0.1,
    model="gpt-4o-mini",
    streaming=True
)

#====================Output format=====================
class RoeIndicator(BaseModel):
    ROE_value : float = Field(description="Rate of ROE")
    ROE_report : str = Field(description="Detailed analysis of ROE")

class PerIndicator(BaseModel):
    PER_value : float = Field(description="Rate of PER")
    PER_report : str = Field(description="Detailed analysis of PER")

class FfoIndicator(BaseModel):
    FFO_trending : str = Field(description="Trending of FFO(Upward, Downward, Hold)")
    FFO_report : str = Field(description="Detailed analysis of FFO")

class NewsComponent(BaseModel):
    title : str = Field(description="Title of News")
    summary : str = Field(description="Summary of News")
    sentiment : str = Field(description="Sentiment of News")
    url : str = Field(description="Link or Url of News")

class StockNews(BaseModel):
    news : List[NewsComponent]

class DividendPayoutRatioReport(BaseModel):
    Dividend_Payout_Ratio : float = Field(description="Rate of Dividend Payout")
    Dividend_Payout_Ratio_report : str = Field(description="Detailed analysis of Dividend Payout Ratio")

class StockPriceReport(BaseModel):
    stock_price_trending : str = Field(description="Trending of Stock Price(Upward, Downward, Hold)")
    stock_price_report : str = Field(description="Detailed Analysis of stock price")

class AnalysisReport(BaseModel):
    company: str = Field(description="Company Name")
    decision : str = Field(description="your recommendation to BUY or SELL the stock")
    ROE : RoeIndicator
    PER : PerIndicator
    FFO : FfoIndicator
    dividend_payout_ratio_report : DividendPayoutRatioReport
    stock_new : StockNews = Field(description="Relating News of Stock. You should contain title, summmary, sentiment, url")
    stock_price_report : StockPriceReport
    recommendation : str = Field(description="Final Comprehensive Recommendation")

class CompanyOverview(BaseModel):
    overview : str = Field(description="Overview Information of Company")
    symbol : str = Field(description="stock symbol of company")

#====================Tools=====================
class Tools:
    def get_ROE(symbol):
        stock = yf.Ticker(symbol)

        try :
            value = str(stock.info['returnOnEquity'] * 100) + "%"
            return f"ROE Rate of {symbol} : {value}"
        
        except KeyError:
            return f"ROE Rate is not Provided"

    def get_PER(symbol):
        stock = yf.Ticker(symbol)

        try :
            value = stock.info['trailingPE']
            return f"PER Rate of {symbol} : {value}"
    
        except KeyError:
            return f"PER Rate is not Provided"
    
    def get_FFO(symbol):
        stock = yf.Ticker(symbol)

        try :
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
        
        except KeyError:
            return f"FFO Rate is not Provided"

    def get_diviend_payout_ratio(symbol):
        stock = yf.Ticker(symbol)

        try :
            value = str(round(stock.info['payoutRatio'] * 100, 2)) + "%"
            return f"Dividend Payout Ratio of {symbol} : {value}"
        
        except KeyError:
            return f"Dividend Payout Ratio is not Provided"
    
    def get_news(symbol):
        stock = yf.Ticker(symbol)
        try :
            return list(map(lambda x: x["link"], stock.news))[0:4]   
        except KeyError:
            return f"News Link is not provided"

    def get_stock_price(symbol): 
        stock = yf.Ticker(symbol)

        try :
            price_history = stock.history(period="1y")['High']
            price_history.index = price_history.index.date

            return price_history
        except KeyError:
            return f"Stock Price is not Provided"

def company_researcher(companys : str):
    search_result = DuckDuckGoSearchAPIWrapper().run(companys)

    output_parser = PydanticOutputParser(pydantic_object=CompanyOverview)

    prompt = ChatPromptTemplate.from_messages([
    ('system', """
        Answer the question using ONLY the following context. Do NOT makeup your own. 
        
        Context : {search_result}

        {format_instructions}
    """),
    ('human', "Tell me about {company} including about stock symbol.")
])

    chain = prompt | llm | output_parser

    result = chain.invoke({'search_result' : search_result, 'company': companys, 'format_instructions': output_parser.get_format_instructions()})

    return result

async def ROE_analyst( context : str, symbol : str):
     
    ROE = await asyncio.to_thread(Tools.get_ROE, symbol)

    ROE_info = "ROE measures the efficiency and profitability of a company's equity investment. A higher ROE indicates more effective management and profitability, with a benchmark of 15% being desirable. Companies with high ROE are often well-managed and show potential for long-term growth."

    prompt = ChatPromptTemplate.from_messages([
        ('system', """
            Answer the question using ONLY the following context. Do NOT makeup your own. 
            
            Company : {symbol}

            Context1 : {context}
            
            Context2 : {ROE_info}
            
            Context3 : {ROE}

        """),
        ('human', "Give me recommendation whether {symbol} is good to invest or not. Give me recommendation in terms of ROE rate.")
    ])

    chain = prompt | llm

    result = await chain.ainvoke({'context': context, 'ROE_info': ROE_info, 'ROE': ROE, 'symbol': symbol})
     
    return result.content

async def PER_analyst( context : str, symbol : str):
     
    PER = await asyncio.to_thread(Tools.get_PER, symbol)

    PER_info = "PER assesses if a stock is undervalued or overvalued by comparing the stock price to its EPS. A PER below 1 suggests the stock may be undervalued, while a PER above 1 indicates possible overvaluation. Investing in stocks with a PER below 1 is generally seen as a wise choice."

    prompt = ChatPromptTemplate.from_messages([
        ('system', """
            Answer the question using ONLY the following context. Do NOT makeup your own. 
            
            Company : {symbol}

            Context1 : {context}
            
            Context2 : {PER_info}
            
            Context3 : {PER}

        """),
        ('human', "Give me recommendation whether {symbol} is good to invest or not. Give me recommendation in terms of PER rate.")
    ])

    chain = prompt | llm

    result = await chain.ainvoke({'context': context, 'PER_info': PER_info, 'PER': PER, 'symbol': symbol})
     
    return result.content

async def FFO_analyst( context : str, symbol : str):
     
    FFO = await asyncio.to_thread(Tools.get_FFO, symbol)

    FFO_info = "FFO reflects earnings from core operations by adjusting net income for non-operational gains and expenses. It's a more accurate measure of a company’s operational performance compared to net income. Companies with consistently growing FFO over the years are considered good investment opportunities."

    prompt = ChatPromptTemplate.from_messages([
        ('system', """
            Answer the question using ONLY the following context. Do NOT makeup your own. 
            
            Company : {symbol}

            Context1 : {context}
            
            Context2 : {FFO_info}
            
            Context3 : {FFO}

        """),
        ('human', "Give me recommendation whether {symbol} is good to invest or not. Give me recommendation in terms of FFO rate.")
    ])

    chain = prompt | llm

    result = await chain.ainvoke({'context': context, 'FFO_info': FFO_info, 'FFO': FFO, 'symbol': symbol})
     
    return result.content

async def dividend_payout_ratio_analyst( context : str, symbol : str):
     
    payout_ratio = await asyncio.to_thread(Tools.get_diviend_payout_ratio, symbol)

    payout_info = "The dividend payout ratio refers to the percentage of a company's net income that is distributed to shareholders in the form of dividends. A higher payout ratio means the company is returning a larger portion of its profits to shareholders, while a lower ratio suggests the company is reinvesting more profits back into its operations. It is generally most advantageous to invest in companies with a dividend payout ratio close to 40%, as this level tends to result in stable dividends in the future."

    prompt = ChatPromptTemplate.from_messages([
        ('system', """
            Answer the question using ONLY the following context. Do NOT makeup your own. 
            
            Company : {symbol}

            Context1 : {context}
            
            Context2 : {payout_info}
            
            Context3 : {payout_ratio}

        """),
        ('human', "Give me recommendation whether {symbol} is good to invest or not. Give me recommendation in terms of dividend payout ratio.")
    ])

    chain = prompt | llm

    result = await chain.ainvoke({'context': context, 'payout_info': payout_info, 'payout_ratio': payout_ratio, 'symbol': symbol})
     
    return result.content

async def news_analyst(context, symbol):
     
    news_url = await asyncio.to_thread(Tools.get_news, symbol)

    parser = PydanticOutputParser(pydantic_object=StockNews)

    prompt = ChatPromptTemplate.from_messages([
        ('system', """
            Answer the question using ONLY the following context. Do NOT makeup your own. 
            
            Company : {symbol}
            Context : {context}
            
            Summarize important news and give insight whether it is good to invest.
            Here is news links you should check.
            News Links : {news_url}
            
            {format_instructions}
        
            """),
    ])

    chain = prompt | llm

    result = await chain.ainvoke({'context': context, 'news_url': news_url, 'format_instructions': parser.get_format_instructions(), 'symbol': symbol})

    return result.content

async def stock_price_analyst(symbol):
     
    price_stock = await asyncio.to_thread(Tools.get_stock_price, symbol)

    prompt = ChatPromptTemplate.from_messages([
        ('system', """
            Answer the question using ONLY the following context. Do NOT makeup your own. 
            
            Company : {symbol}
            
            1 Year's Stock Price
            {price_stock}

        """),
        ('human', "Give me recommendation whether {symbol} is good to invest or not. Give me recommendation in terms of stock price")
    ])

    chain = prompt | llm

    result = await chain.ainvoke({'price_stock': price_stock, 'symbol': symbol})

     
    return result.content

async def hedge_fund_manager(company):
    result = company_researcher(company)

    company_overview = result.overview
    company_symbol = result.symbol

    information = await asyncio.gather(
        ROE_analyst(company_overview, company_symbol),
        PER_analyst(company_overview, company_symbol),
        FFO_analyst(company_overview, company_symbol),
        dividend_payout_ratio_analyst(company_overview, company_symbol),
        news_analyst(company_overview, company_symbol),
        stock_price_analyst(company_symbol)
    )

    parser = PydanticOutputParser(pydantic_object=AnalysisReport)

    prompt = ChatPromptTemplate.from_messages([
        ('system', """
            Provide a detailed investment recommendation for {symbol}'s stock. Include your analysis of the stock's potential risks and rewards, and provide a clear rationale for your recommendation
            Your final answer MUST be a detailed investment recommendation report to BUY or SELL the stock that includes your analysis of the stock's potential risks and rewards. Provide a clear rationale for your recommendation ONLY based on following information.

            The importance of each piece of information varies. 
            The importance decreases from left to right, meaning the priority of information is as follows three indicator(ROE, PER, FFO) > dividend payout ratio > news > stock price. 

            Following Contexts                
            Company : {symbol}
            
            Company Overview : {overview}

            ROE : {ROE}
            
            PER : {PER}
            
            FFO : {FFO}
            
            Payout Ratio : {payout_ratio}

            News : {news_analysis}
            (You should get minimum three articles)
            
            Stock Price : {price_analysis}
            
            {format_instructions}

        """),
        ('human', "Give me recommendation whether {symbol} is good to invest or not.")
    ])

    chain = prompt | llm

    result = chain.invoke({'overview': company_overview, 'symbol': company_symbol, 'ROE': information[0], 'PER': information[1], 'FFO': information[2], 'payout_ratio': information[3], 'news_analysis': information[4], 'price_analysis': information[5], 'format_instructions': parser.get_format_instructions()})

    return result.content


@st.cache_resource(show_spinner="Gather Information...")
def kick(company):
    text = asyncio.run(hedge_fund_manager(company))
    text = text.replace("```json", "").replace("```", "")

    return text

def new_formattor(result):
    new_text = list()
    for new in result['stock_new']['news']:
        new_text.append(
            f"""<li>
            <strong>{new['title']}</strong>
            <ul>
                <li><strong>Summary</strong> : {new['summary']}</li>
                <li><strong>Sentiment</strong> : {new['sentiment']}</li>
                <li><a href={new['url']} target=_blank style="text-decoration : none; color: #00acee">Link</a></li>
            </ul>
            </li>""")
    return '<br/>'.join(new_text)

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
    result = kick(stock)
    result = json.loads(result)

    news = new_formattor(result).replace("`", "")

    st.markdown(
        f"""

        <h1>Stock Recommendation for {result['company']}</h1>

        <h2>Recommendation : {result['decision']}</h2>

        <h3>Indicator</h3>

        - **ROE(Return on Equity)**
            - {result['ROE']['ROE_value']}%
            - {result['ROE']['ROE_report']}
        
        - **PER(Price to Earning Ratio)**
            - {result['PER']['PER_value']}
            - {result['PER']['PER_report']}

        - **FFO(Fund from operations)**
            - {result["FFO"]['FFO_trending']}
            - {result['FFO']['FFO_report'].replace("$", "")}

        <h3>Dividend Payout Ratio</h3>

        - Payout Ratio : **{result['dividend_payout_ratio_report']['Dividend_Payout_Ratio']}%**
        - {result['dividend_payout_ratio_report']['Dividend_Payout_Ratio_report'].replace("$", "")}

        <h3>Company News</h3>

        <ul>
            {news}
        </ul>

        <h3>Stock Price</h3>

        - Stock Price Trending : {result['stock_price_report']['stock_price_trending'].replace("$", "")}
        - {result['stock_price_report']['stock_price_report'].replace("$", "")}

        ### Conclusion
        - {result['recommendation'].replace("$", "")}


        """, unsafe_allow_html=True)
