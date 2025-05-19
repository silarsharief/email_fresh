from jinja2 import Template
import datetime
import os
import yfinance as yf
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.patheffects import withStroke
import json
from langchain_groq import ChatGroq
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv
import matplotlib.patches as patches
import matplotlib.patheffects as pe
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def plot_market_data(df, date_str, index_name, output_file):
    # Set up fonts with thicker weights
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    # Create figure and axis with transparent background
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0)  # Keep figure background fully transparent
    ax.set_facecolor('none')  # No axes background

    # Determine line color based on trend
    if df['Close'].iloc[-1] > df['Close'].iloc[0]:
        line_color = '#388e3c'  # Green
    else:
        line_color = '#c62828'  # Red

    # Plot the data with high-contrast color
    line, = ax.plot(df.index, df['Close'], linewidth=2, color=line_color)
    
    # Customize the graph
    ax.set_title(index_name, fontsize=16, pad=10, color='grey', weight='bold')
    
    # Add date with custom font
    date_font = font_manager.FontProperties(family='monospace', weight='bold', size=12)
    ax.text(0.5, 0.95, date_str, transform=ax.transAxes, 
            horizontalalignment='center', color=line_color,
            fontproperties=date_font, alpha=0.8)
    
    # Remove x-axis labels and ticks
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Format y-axis with minimal ticks and grey color
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Show only 4 y-axis ticks
    plt.yticks(color='#808080', fontsize=10, weight='bold')  # Grey color for y-axis labels
    
    # Add minimal grid with grey color
    ax.grid(True, linestyle='--', alpha=0.3, color='#808080')
    
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add text shadow effect
    for text in ax.texts:
        text.set_path_effects([withStroke(linewidth=2, foreground='black')])
    
    # Save the plot with transparent background
    plt.savefig(os.path.join(script_dir, output_file), dpi=300, bbox_inches='tight', 
                transparent=True)
    plt.close()






def get_market_data():
    try:
        # Get today's date
        today = datetime.now().date()
        start_time = datetime.combine(today, datetime.min.time())
        end_time = datetime.now()
        
        # Fetch Nifty 50 data
        nifty = yf.Ticker("^NSEI")
        nifty_df = nifty.history(start=start_time, end=end_time, interval="1m")
        if not nifty_df.empty:
            plot_market_data(nifty_df, today.strftime('%Y-%m-%d'), 'NIFTY 50', 'nifty50_graph.png')
        
        # Fetch Sensex data
        sensex = yf.Ticker("^BSESN")
        sensex_df = sensex.history(start=start_time, end=end_time, interval="1m")
        if not sensex_df.empty:
            plot_market_data(sensex_df, today.strftime('%Y-%m-%d'), 'SENSEX', 'sensex_graph.png')
            
        return True
            
    except Exception as e:
        print(f"Error generating market graphs: {str(e)}")
        return False

def get_nifty_data():
    try:
        # Get Nifty 50 data
        nifty = yf.Ticker("^NSEI")
        
        # Get today's date
        today = datetime.now().date()
        start_time = datetime.combine(today, datetime.min.time())
        end_time = datetime.now()
        
        # Fetch 1-minute data
        df = nifty.history(start=start_time, end=end_time, interval="1m")
        
        if not df.empty:
            # Create and save the graph
            plot_market_data(df, today.strftime('%Y-%m-%d'), 'NIFTY 50', 'nifty50_graph.png')
            return True
        return False
            
    except Exception as e:
        print(f"Error generating Nifty graph: {str(e)}")
        return False

def get_market_status():
    # Get current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    # Check if it's a weekday
    if current_time.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        return False, "Market is closed (Weekend)"
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_start = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if current_time < market_start:
        return False, f"Market opens at {market_start.strftime('%I:%M %p')} IST"
    elif current_time > market_end:
        return False, f"Market closed at {market_end.strftime('%I:%M %p')} IST"
    else:
        return True, "Market is open"

def get_nifty50_symbols():
    # Nifty 50 symbols with .NS suffix for NSE (removed HDFC as it's delisted)
    nifty50_symbols = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'ULTRACEMCO.NS',
        'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS',
        'NTPC.NS', 'HCLTECH.NS', 'TECHM.NS', 'WIPRO.NS', 'SUNPHARMA.NS',
        'TATAMOTORS.NS', 'BRITANNIA.NS', 'SHREECEM.NS', 'JSWSTEEL.NS', 'TATACONSUM.NS',
        'BPCL.NS', 'INDUSINDBK.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
        'HEROMOTOCO.NS', 'GRASIM.NS', 'CIPLA.NS', 'UPL.NS', 'BAJAJFINSV.NS',
        'TATASTEEL.NS', 'ADANIPORTS.NS', 'M&M.NS', 'HDFCLIFE.NS', 'DIVISLAB.NS',
        'IOC.NS', 'SBILIFE.NS', 'TATAPOWER.NS', 'HINDALCO.NS'
    ]
    return nifty50_symbols

def get_sensex_symbols():
    # Sensex symbols with .NS suffix for NSE (same as Nifty 50 for better data availability)
    sensex_symbols = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'ULTRACEMCO.NS',
        'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS',
        'NTPC.NS', 'HCLTECH.NS', 'TECHM.NS', 'WIPRO.NS', 'SUNPHARMA.NS',
        'TATAMOTORS.NS', 'BRITANNIA.NS', 'SHREECEM.NS', 'JSWSTEEL.NS', 'TATACONSUM.NS'
    ]
    return sensex_symbols

def get_index_change(index_symbol, index_name):
    try:
        # Get current time in IST
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        
        # Get the ticker info directly
        ticker = yf.Ticker(index_symbol)
        info = ticker.info
        
        # Get the current price and change percentage
        current_price = info.get('regularMarketPrice', 0)
        change_percent = info.get('regularMarketChangePercent', 0)
        
        # Get market status
        is_market_open, status = get_market_status()
        
        print(f"\nFetching {index_name} data:")
        print(f"Current price: {current_price}")
        print(f"Change percent: {change_percent}")
        print(f"Market status: {status}")
        
        if current_price > 0:
            return {index_name: [round(change_percent, 2), round(current_price, 2)]}
        else:
            print(f"Invalid price data for {index_name}")
            return {index_name: ["Invalid price data", "Invalid price data"]}
            
    except Exception as e:
        print(f"Error fetching {index_name} data: {str(e)}")
        return {index_name: ["Error fetching data", "Error fetching data"]}

def get_nifty_movers():
    try:
        # Get current time in IST
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        
        # Get market status
        is_market_open, status = get_market_status()
        print(f"\nMarket status: {status}")
        
        # Get the top gainers and losers
        gainers = {}
        losers = {}
        
        # Get detailed info for each stock in Nifty 50
        for symbol in get_nifty50_symbols():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                current_price = info.get('regularMarketPrice', 0)
                change_percent = info.get('regularMarketChangePercent', 0)
                
                if current_price > 0:
                    company_name = symbol.replace('.NS', '')
                    stock_data = {
                        'price': round(current_price, 2),
                        'change': round(change_percent, 2)
                    }
                    
                    # Sort into gainers or losers
                    if change_percent > 0:
                        gainers[company_name] = stock_data
                    else:
                        losers[company_name] = stock_data
                        
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Sort and get top 5
        sorted_gainers = dict(sorted(gainers.items(), key=lambda x: x[1]['change'], reverse=True)[:5])
        sorted_losers = dict(sorted(losers.items(), key=lambda x: x[1]['change'])[:5])
        
        # Print for verification
        print("\nTop Gainers:")
        for name, data in sorted_gainers.items():
            print(f"{name}: {data['change']}% (₹{data['price']})")
        print("\nTop Losers:")
        for name, data in sorted_losers.items():
            print(f"{name}: {data['change']}% (₹{data['price']})")
            
        return sorted_gainers, sorted_losers
    except Exception as e:
        print(f"Error fetching market movers data: {str(e)}")
        # Return empty dictionaries instead of None to prevent NoneType errors
        return {}, {}

def get_news_for_tickers(ticker_list):
    """
    Fetch news for a list of stock tickers and return in JSON format
    """
    news_data = {
        "ticker_list": ticker_list,
        "news_data": {}
    }
    
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            if news:  # Only store if we got news
                news_data["news_data"][ticker] = []
                for article in news:
                    try:
                        # Access the nested content structure
                        content = article.get('content', {})
                        
                        # Extract information from content
                        title = content.get('title', 'No title available')
                        summary = content.get('summary', 'No summary available')
                        pub_date = content.get('pubDate', 'No date available')
                        
                        # Get provider information
                        provider = content.get('provider', {})
                        publisher = provider.get('displayName', 'Unknown publisher')
                        
                        # Get URL information
                        canonical_url = content.get('canonicalUrl', {})
                        link = canonical_url.get('url', 'No link available')
                        
                        # Convert timestamp to readable format if available
                        if pub_date != 'No date available':
                            try:
                                pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                pub_date = pub_date.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                pass
                        
                        # Create article dictionary
                        article_data = {
                            "title": title,
                            "summary": summary,
                            "publisher": publisher,
                            "link": link,
                            "published_date": pub_date
                        }
                        
                        news_data["news_data"][ticker].append(article_data)
                        
                    except Exception as e:
                        print(f"Error processing article for {ticker}: {str(e)}")
            else:
                news_data["news_data"][ticker] = []
        except Exception as e:
            news_data["news_data"][ticker] = []
            print(f"Error fetching news for {ticker}: {str(e)}")
    
    return news_data

def summarize_news(news_data):
    # Initialize LangChain with the appropriate model
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    system_prompt = SystemMessagePromptTemplate.from_template(
        """You are a financial news analyst. Create a concise, engaging summary of the most important stock news. Focus on key insights and market-moving information. Format as 4 distinct points with proper spacing between each point.

Rules:
- No introductory text or bullet points
- Each point should be self-contained and impactful
- Use 1-2 relevant emojis per point
- Keep total length around 75 words
- Focus on the most market-relevant news
- Use active voice and present tense
- Make it engaging but professional

Example format:
Tata Steel: investing $1.56 billion to upgrade its Port Talbot site 🏭

Hindalco: to invest $10 billion to expand its domestic business 💰

Hero MotoCorp: reported a 6.4% increase in profit 📈

Powergrid Corp: showcased robust profits in Q3 ⚡"""
    )

    human_prompt = HumanMessagePromptTemplate.from_template(
        "Here is today's stock news in JSON format:\n\n{json_data}\n\nSummarize this as a professional stock market newsletter. Keep it to 4 key points with proper spacing between each point."
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = prompt | llm 

    formatted_output = chain.invoke({"json_data": json.dumps(news_data, indent=2)})

    return formatted_output

# Get overall index changes
print("\nFetching market data...")
nifty_change = get_index_change("^NSEI", "Nifty 50")
sensex_change = get_index_change("^BSESN", "Sensex")
market_changes = {**nifty_change, **sensex_change}

# Get Nifty 50 movers
gainers, losers = get_nifty_movers()
all_tickers = [f"{symbol}.NS" for symbol in gainers.keys()] + [f"{symbol}.NS" for symbol in losers.keys()]

# Print the values for debugging
print("\nMarket Changes:")
print(f"Nifty 50: {nifty_change['Nifty 50']}")
print(f"Sensex: {sensex_change['Sensex']}")

# Helper to get direction, color, and symbol
def get_direction_info(change):
    if isinstance(change, (int, float)):
        if change > 0:
            return {'symbol': '▲', 'color': '#2e7d32', 'direction': 'up'}
        elif change < 0:
            return {'symbol': '▼', 'color': '#c62828', 'direction': 'down'}
    return {'symbol': '', 'color': '#6c757d', 'direction': 'neutral'}

nifty_info = get_direction_info(nifty_change['Nifty 50'][0])
sensex_info = get_direction_info(sensex_change['Sensex'][0])

# Get news data
news_for_tickers = get_news_for_tickers(all_tickers)
news_summary = summarize_news(news_for_tickers)

# Generate market graphs
get_market_data()


import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from urllib.parse import urljoin
import time

async def extract_json_ld(session, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Add headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Make the request with increased timeout
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find the JSON-LD script tag
                    script_tag = soup.find('script', {'type': 'application/ld+json'})
                    
                    if script_tag:
                        try:
                            # Parse the JSON data
                            json_data = json.loads(script_tag.string)
                            return json_data
                        except json.JSONDecodeError:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2)  # Wait before retry
                                continue
                            return None
                elif response.status == 429:  # Too Many Requests
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)  # Exponential backoff
                        await asyncio.sleep(wait_time)
                        continue
                return None
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None
    return None

async def process_company(session, row):
    company_name = row['Company Name']
    url = row['Link']
    
    if pd.notna(url):
        json_data = await extract_json_ld(session, url)
        
        if json_data:
            # Extract relevant information
            company_info = {
                'Company Name': company_name,
                'URL': url,
                'Share Price': None,
                'Face Value': None,
                'ISIN': None,
                'Market Cap': None,
                'P/E Ratio': None,
                'Debt/Equity Ratio': None,
                'ROE%': None,
                'Book Value': None,
                'CIN': None,
                'PAN': None,
                'Rating': None
            }
            
            # Extract properties from additionalProperty
            if 'additionalProperty' in json_data:
                for prop in json_data['additionalProperty']:
                    name = prop.get('name')
                    value = prop.get('value')
                    if name in company_info:
                        company_info[name] = value
            
            # Extract rating if available
            if 'aggregateRating' in json_data:
                company_info['Rating'] = json_data['aggregateRating'].get('ratingValue')
            
            return company_info
    return None

async def process_all_links(urls=None):
    try:
        # Use provided URLs or default list
        if urls is None:
            urls = [
                "https://unlistedzone.com/shares/nsdl-unlisted-shares",
                "https://unlistedzone.com/shares/nse-india-limited-unlisted-shares",
                "https://unlistedzone.com/shares/orbis-financial-corporation-unlisted-shares",
                "https://unlistedzone.com/shares/msei-share-price-buy-sell-unlisted-shares-of-msei-metropolitan-stock-exchange",
                "https://unlistedzone.com/shares/hero-fincorp-limited-share-price-buy-sell-unlisted-shares-of-hero-fincorp",
                "https://unlistedzone.com/shares/csk-share-price-buy-sell-unlisted-shares",
                "https://unlistedzone.com/shares/buy-sell-share-price-aspire-home-finance-corporation-unlisted-shares",
                "https://unlistedzone.com/shares/hdfc-securities-limited-share-price-buy-sell-hdfc-securities-unlisted-shares",
                "https://unlistedzone.com/shares/tata-capital-limited-unlisted-share",
                "https://unlistedzone.com/shares/indian-potash-limited-unlisted-share-buy-sell-share-price",
                "https://unlistedzone.com/shares/oravel-stays-limited-oyo-unlisted-shares"
            ]
        
        total_companies = len(urls)
        
        print(f"Starting async processing of {total_companies} companies...")
        
        # Configure aiohttp session with connection pooling and limits
        connector = aiohttp.TCPConnector(limit=5, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=60)
        
        # Create a session for all requests
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for all links
            tasks = [extract_json_ld(session, url) for url in urls]
            
            # Process all tasks with progress bar
            results = []
            processed = 0
            successful = 0
            
            # Create progress bar
            pbar = tqdm(total=total_companies, desc="Processing companies", unit="company")
            
            for task in asyncio.as_completed(tasks):
                result = await task
                processed += 1
                if result:
                    results.append(result)
                    successful += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Processed': processed,
                    'Successful': successful,
                    'Success Rate': f"{(successful/processed)*100:.1f}%"
                })
            
            pbar.close()
            
            # Return a dictionary with results and statistics
            return {
                'results': results,
                'statistics': {
                    'total_processed': processed,
                    'successful': successful,
                    'success_rate': (successful/processed)*100 if processed > 0 else 0
                }
            }
    except Exception as e:
        return {
            'error': str(e),
            'results': [],
            'statistics': {
                'total_processed': 0,
                'successful': 0,
                'success_rate': 0
            }
        }


data = asyncio.run(process_all_links())


"""
if __name__ == "__main__":
    # Example usage
    results = asyncio.run(process_all_links())
    print("\nProcessing complete!")
    print(f"Total companies processed: {results['statistics']['total_processed']}")
    print(f"Successfully extracted data: {results['statistics']['successful']}")
    print(f"Success rate: {results['statistics']['success_rate']:.1f}%")
    print("\nExtracted Data:")
    for result in results['results']:
        print(json.dumps(result, indent=4, ensure_ascii=False)) """




import math
import json



# Convert additionalProperty list to a dictionary
def convert_properties_to_dict(prop_list):
    return {item["name"]: item["value"] for item in prop_list}

# Main scoring function
def calculate_score(company):
    props = convert_properties_to_dict(company["additionalProperty"])
    try:
        roe = float(props["ROE%"].strip())
        pe_ratio = float(props["P/E Ratio"]) if props["P/E Ratio"] != "N/A" else 100
        debt_equity = float(props["Debt/Equity Ratio"])
        book_value = float(props["Book Value"].replace("₹", "").replace(",", ""))
        price = float(props["Share Price"].replace("₹", "").replace(",", ""))
        market_cap = float(props["Market Cap"].replace("₹", "").replace(",", ""))
        rating = float(company["aggregateRating"]["ratingValue"])
    except Exception as e:
        print(f"Error parsing data for {company['name']}: {e}")
        return -1, {}

    score = 0
    score += roe * 2
    score += (5 if pe_ratio == 0 else 50 / pe_ratio)
    score += (10 if debt_equity == 0 else 10 / debt_equity)
    score += (book_value / price) * 10
    score += math.log10(market_cap + 1) * 1
    score += rating * 5

    metrics = {
        "Name": company["name"],
        "Score": round(score, 2),
        "ROE%": roe,
        "P/E Ratio": pe_ratio,
        "Debt/Equity Ratio": debt_equity,
        "Book Value": book_value,
        "Price": price,
        "Book Value/Price": round(book_value / price, 2),
        "Market Cap": market_cap,
        "Rating": rating
    }
    return score, metrics

# Process all companies
results = []
for company in data['results']:  # Access the 'results' key
    score, metrics = calculate_score(company)
    if score >= 0:
        results.append(metrics)

# Sort results by score
results.sort(key=lambda x: x["Score"], reverse=True)

# Print
list_of_companies = []
for r in results:
    name = r['Name'].split(' Unlisted Shares Price')[0]
    pe = r['P/E Ratio']
    price = r['Price']
    dct = {'name': name, 'price': price, 'pe': pe}
    list_of_companies.append(dct)

# Sort by PE ratio (lower is better) and take top 5
list_of_companies.sort(key=lambda x: float(x['pe']) if isinstance(x['pe'], (int, float)) else float('inf'))
list_of_companies = list_of_companies[:5]

# Sample data
data = {
    'title': 'TruWealth.club Market Digest',
    'date': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%B %d, %Y'),
    'nifty': nifty_change['Nifty 50'][1],
    'nifty_change': nifty_change['Nifty 50'][0],
    'nifty_symbol': nifty_info['symbol'],
    'nifty_color': nifty_info['color'],
    'sensex': sensex_change['Sensex'][1],
    'sensex_change': sensex_change['Sensex'][0],
    'sensex_symbol': sensex_info['symbol'],
    'sensex_color': sensex_info['color'],
    'gainers': [{'name': k, 'price': v['price'], 'change': v['change']} for k, v in gainers.items()],
    'losers': [{'name': k, 'price': v['price'], 'change': v['change']} for k, v in losers.items()],
    'news_summary': news_summary.content if news_summary else "No news summary available",
    'pre_ipo': list_of_companies
}

# Read the template file
template_path = os.path.join(script_dir, 'test.html')
with open(template_path, 'r', encoding='utf-8') as file:
    template_content = file.read()

# Create template object
template = Template(template_content)

# Render the template
rendered_html = template.render(**data)

# Save the rendered HTML in the same directory as the script
output_path = os.path.join(script_dir, 'rendered_email.html')
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(rendered_html)

print(f"Email template has been rendered to '{output_path}'") 