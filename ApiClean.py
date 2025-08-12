import feedparser
import yfinance as yf

from datetime import datetime, timedelta
from newsapi import NewsApiClient

import requests
from dotenv import load_dotenv
import os
import re
import pandas as pd
import time
from transformers import pipeline


def fetch_finnhub_news(ticker):
    load_dotenv()  # Load environment variables from .env file

    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    if not FINNHUB_API_KEY:
        print("Finnhub API key is missing.")
        return []

    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = []

        for item in data[:20]:
            raw_title = item.get('headline', '')
            raw_content = item.get('summary', '')

            title = raw_title.lower() if raw_title else ''
            content = raw_content.lower() if raw_content else ''
            ticker_lower = ticker.lower()

            # Only include if ticker is found in title or content
            if ticker_lower in title or ticker_lower in content:
                try:
                    readable_date = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M:%S')
                except (KeyError, TypeError, ValueError):
                    readable_date = "Unknown"

                articles.append({
                    'date': readable_date,
                    'title': raw_title,
                    'content': raw_content,
                    'source': 'Finnhub',
                    'url': item.get('url', '')
                })

            if len(articles) == 20:
                break

        return articles

    except requests.RequestException as e:
        print(f"Error fetching Finnhub news for {ticker}: {str(e)}")
        return []
    

def is_relevant(title, content, ticker):
    title = str(title) if title is not None else ""
    content = str(content) if content is not None else ""
    text = (title + " " + content).lower()
    ticker = ticker.lower()
    return ticker in text or 'stock' in text or 'share' in text or 'market' in text




def clean_text(text):
    """Clean and preprocess the text"""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def fetch_newsapi_news(ticker):
    load_dotenv()
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    if not NEWS_API_KEY:
        print("NewsAPI key is missing.")
        return []

    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    try:
        query = f'"{ticker}" AND (stock OR shares OR earnings OR finance OR market)'

        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        response = newsapi.get_everything(q=query,
                                          language='en',
                                          sort_by='publishedAt',
                                          from_param=from_date,
                                          to=to_date,
                                          page_size=20)

        articles = []

        for item in response['articles']:
            raw_title = item.get('title', '')
            raw_description = item.get('description', '')

            title = raw_title.lower() if raw_title else ''
            content = raw_description.lower() if raw_description else ''

            # Define keyword list including ticker
            keywords = [
                'stock', 'share', 'shares', 'earnings', 'revenue', 'profit', 'loss',
                'acquisition', 'merger', 'ipo', 'buyback', 'dividend', 'forecast', 'guidance',
                'market', 'finance', 'financial', 'investment', 'investor', 'trading',
                'company', 'results', 'quarter', 'report', 'business', 'valuation', 'layoffs',
                'sec', 'ceo', 'cfo', 'announcement', 'strategy', 'expansion',
                ticker.lower()
            ]

            # OR condition: ticker in title/content OR any keyword in title/content
            if ticker.lower() in content or ticker.lower() in title or \
               any(kw in content or kw in title for kw in keywords):

                try:
                    published_at = datetime.strptime(item['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    formatted_date = published_at.strftime('%Y-%m-%d %H:%M:%S')
                except (KeyError, ValueError):
                    formatted_date = "Unknown"

                # Optional: Clean text if you have a clean_text() function
                cleaned_title = clean_text(raw_title) if raw_title else ''
                cleaned_content = clean_text(raw_description) if raw_description else ''

                articles.append({
                    'ticker': ticker,
                    'date': formatted_date,
                    'title': raw_title,
                    'content': raw_description,
                    'cleaned_title': cleaned_title,
                    'cleaned_content': cleaned_content,
                    'source': 'NewsAPI',
                    'url': item.get('url', '')
                })

            if len(articles) == 20:
                break

        return articles

    except Exception as e:
        print(f"Error fetching NewsAPI news for {ticker}: {str(e)}")
        return []
def collect_and_preprocess_data(ticker):
    all_data = []
    print(f"Fetching news for {ticker}...")
    newsapi_articles = fetch_newsapi_news(ticker)
    finnhub_articles = fetch_finnhub_news(ticker)
    all_articles = newsapi_articles + finnhub_articles
    if not all_articles:
        print(f"No articles found for {ticker}")
        return []
    for article in all_articles:
        title = article.get('title')
        content = article.get('content')
        date = article.get('date')
        source = article.get('source')
        url = article.get('url')

        # Skip articles with missing essential data
        if not all([title, content, date, source, url]):
            print(f"Skipping article with missing data for {ticker}")
            continue

        if not is_relevant(title, content, ticker):
            continue
        cleaned_title = clean_text(title)
        cleaned_content = clean_text(content)
        all_data.append({
            'ticker': ticker,
            'date': date,
            'title': title,
            'content': content,
            'cleaned_title': cleaned_title,
            'cleaned_content': cleaned_content,
            'source': source,
            'url': url
        })
        # Add a small delay to avoid overwhelming servers
        time.sleep(0.1)
    return all_data



def article_content_sentiment(stock_data):
    pipe = pipeline("text-classification", model="ProsusAI/finbert")  # Load pretrained model

    def sentiment_score(sentiment):
        label = sentiment['label'].lower()
        if label == 'positive':
            return 1
        elif label == 'negative':
            return -1
        else:  # neutral or other
            return 0

    avg_score = 0
    article_titles = []
    for article in stock_data:
        cleaned_title = article['cleaned_title']
        article_titles.append(cleaned_title)
        cleaned_content = article['cleaned_content']

        sentiment_title = pipe(cleaned_title)[0]
        sentiment_content = pipe(cleaned_content)[0]

        title_sent = sentiment_score(sentiment_title)
        content_sent = sentiment_score(sentiment_content)

        avg_article_score = (title_sent + content_sent) / 2

        # print(f"{cleaned_title} Title sentiment: {title_sent}\n")
        # print(f"{cleaned_content} Content sentiment: {content_sent}\n")
        # print(f"Average article sentiment: {avg_article_score}\n")

        avg_score += avg_article_score

    overall_avg_score = avg_score / len(stock_data) if stock_data else 0
    # print(f"Overall average sentiment score: {overall_avg_score}")
    return overall_avg_score , article_titles


def run():
    ticker_or_title = input("Enter Stock Ticker (Or company name (ticker more accurate)): ")
    stock_data = collect_and_preprocess_data(ticker_or_title)
    
    final_score, article_titles = article_content_sentiment(stock_data)
    
    return {
        "ticker": ticker_or_title,
        "score": final_score,
        "news_titles": article_titles
    }
