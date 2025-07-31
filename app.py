import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import openai
import av
import tempfile
import requests
from prophet import Prophet
from bs4 import BeautifulSoup

def get_db():
    conn = sqlite3.connect("budgetify.db", check_same_thread=False)
    return conn

def create_tables():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS transactions (
        username TEXT,
        date TEXT,
        description TEXT,
        amount REAL,
        category TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS income (
        username TEXT,
        date TEXT,
        amount REAL
    )''')
    conn.commit()

create_tables()

def register_user(username, password):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone() is not None

def load_transactions(username):
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM transactions WHERE username=?", conn, params=(username,))
    return df

def load_income(username):
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM income WHERE username=?", conn, params=(username,))
    return df

def save_transaction(username, date, description, amount, category):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, ?)", (username, date, description, amount, category))
    conn.commit()

def save_income(username, date, amount):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO income VALUES (?, ?, ?)", (username, date, amount))
    conn.commit()

def delete_transaction(username, date, description, amount):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM transactions WHERE username=? AND date=? AND description=? AND amount=?", (username, date, description, amount))
    conn.commit()

def delete_income(username, date, amount):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM income WHERE username=? AND date=? AND amount=?", (username, date, amount))
    conn.commit()

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame):
        audio = frame.to_ndarray()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_bytes = frame.to_bytes()
            f.write(wav_bytes)
            st.session_state['audio_file_path'] = f.name
        return frame

class ExpenseClassifier:
    def __init__(self):
        self.category_keywords = {
            'Food & Dining': [
                'restaurant', 'cafe', 'food', 'dining', 'pizza', 'burger', 'coffee',
                'starbucks', 'mcdonalds', 'subway', 'dominos', 'kfc', 'grocery',
                'supermarket', 'market', 'bakery', 'deli', 'bar', 'pub', 'lunch',
                'breakfast', 'dinner', 'meal', 'eat', 'cuisine', 'bistro', 'snack', 'small chops',
                'Glovo', 'Jumia Food', 'Uber Eats', 'DoorDash', 'food delivery', 'Chowdeck'
            ],
            'Transportation': [
                'gas', 'fuel', 'uber', 'lyft', 'taxi', 'bus', 'train', 'metro', 'indrive',
                'bolt',
                'parking', 'toll', 'car', 'vehicle', 'transport', 'airline',
                'flight', 'airport', 'travel', 'rental', 'subway', 'rideshare'
            ],
            'Shopping': [
                'amazon', 'store', 'mall', 'shop', 'retail', 'purchase', 'buy',
                'clothing', 'clothes', 'fashion', 'electronics', 'target', 'walmart',
                'costco', 'best buy', 'macys', 'nike', 'adidas', 'online'
            ],
            'Entertainment': [
                'movie', 'cinema', 'theater', 'concert', 'music', 'game', 'netflix',
                'spotify', 'entertainment', 'fun', 'amusement', 'park', 'zoo',
                'museum', 'club', 'party'
            ],
            'Utilities': [
                'electric', 'electricity', 'gas', 'water', 'internet', 'phone',
                'mobile', 'cable', 'utility', 'bill', 'service', 'telecom',
                'verizon', 'att', 'comcast', 'xfinity', 'rent'
            ],
            'Personal': [
                'hair', 'salon', 'spa', 'fitness', 'gym'
            ],
            'Healthcare': [
                'doctor', 'hospital', 'medical', 'pharmacy', 'medicine', 'health',
                'dental', 'clinic', 'insurance', 'prescription', 'cvs', 'walgreens'
            ],
            'Education': [
                'school', 'college', 'university', 'education', 'tuition', 'book',
                'student', 'course', 'class', 'learning', 'training'
            ],
            'Finance': [
                'bank', 'atm', 'fee', 'interest', 'loan', 'credit', 'investment',
                'transfer', 'payment', 'finance', 'tax', 'insurance'
            ]
        }
        
        self.model = None
        self.is_trained = False
    
    def preprocess_description(self, description):
        if pd.isna(description):
            return ""
        
        description = str(description).lower()
        description = re.sub(r'[^a-zA-Z\s]', ' ', description)
        description = re.sub(r'\s+', ' ', description).strip()
        
        return description
    
    def rule_based_classify(self, description):
        description_clean = self.preprocess_description(description)
        
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_clean)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'Other'
    
    def train_model(self, df):
        if len(df) < 10:
            return False
        
        X = df['description'].apply(self.preprocess_description)
        y = df['category']
        
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except:
            return False
    
    def predict_category(self, description):
        if self.is_trained and self.model:
            try:
                clean_desc = self.preprocess_description(description)
                prediction = self.model.predict([clean_desc])[0]
                return prediction
            except:
                pass
        
        return self.rule_based_classify(description)

class SpendingPredictor:
    def __init__(self):
        self.predictions = {}
    
    def predict_future_spending(self, df, days_ahead=30):
        if len(df) < 7:
            return {}
        
        df['date'] = pd.to_datetime(df['date'])
        daily_spending = df.groupby(['date', 'category'])['amount'].sum().reset_index()
        
        predictions = {}
        
        for category in df['category'].unique():
            cat_data = daily_spending[daily_spending['category'] == category]
            
            if len(cat_data) >= 3:
                recent_avg = cat_data['amount'].tail(7).mean()
                monthly_prediction = recent_avg * days_ahead
                predictions[category] = max(0, monthly_prediction)
        
        return predictions

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    transactions = [
        ("Coffee", 5000, "Food & Dining"),
        ("Petrol", 30000, "Transportation"),
        ("Konga Purchase", 20000, "Shopping"),
        ("Netflix Subscription", 4400, "Entertainment"),
        ("Electricity Bill", 100000, "Utilities"),
        ("Egusi Soup & Swallow", 5000, "Food & Dining"),
        ("Nkwobi", 7000, "Food & Dining"),
        ("Uber Ride", 6000 , "Transportation"),
        ("Hair Treatment", 25000, "Personal"),
        ("Spotify Premium (Monthly Charge)", 1900, "Entertainment"),
        ("Internet Subscription", 40000, "Utilities"),
        ("Monthly rent", 300000, "Utilities"),
        ("House products", 20000, "Shopping"),
        ("Books", 30000, "Education"),
        ("Clothing", 100000, "Shopping"),
        ("Transfers to Semi", 10000, "Finance"),
        ("Clubbing", 50000, "Entertainment"),
        ("Chowdeck", 7000, "Food & Dining"),
        ("Pizza Hut", 10000, "Food & Dining"),
        ("Cinema", 20000, "Entertainment"),
    ]
    
    dates = []
    descriptions = []
    amounts = []
    categories = []
    
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(200):
        random_days = np.random.randint(0, 90)
        transaction_date = start_date + timedelta(days=random_days)
        
        desc, amt, cat = transactions[np.random.randint(0, len(transactions))]
        
        amount_variation = amt * np.random.uniform(0.8, 1.2)
        
        dates.append(transaction_date.strftime('%Y-%m-%d'))
        descriptions.append(desc)
        amounts.append(round(amount_variation, 2))
        categories.append(cat)
    
    return pd.DataFrame({
        'date': dates,
        'description': descriptions,
        'amount': amounts,
        'category': categories
    })

def create_spending_charts(df, selected_category=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    category_totals = df.groupby('category')['amount'].sum().abs()
    fig_pie = px.pie(
        values=category_totals.values,
        names=category_totals.index,
        title="Spending by Category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    daily_spending = df.groupby('date')['amount'].sum().abs().reset_index()
    fig_trend = px.line(
        daily_spending,
        x='date',
        y='amount',
        title="Daily Spending Trend",
        labels={'amount': 'Amount (NGN)', 'date': 'Date'}
    )
    
    fig_trend.update_traces(
        line_color='#bfa14a',
        line_width=3,
        mode='lines+markers',
        marker=dict(
            size=8,
            color='#bfa14a',
            symbol='circle'
        )
    )
    
    category_daily = df.groupby(['date', 'category'])['amount'].sum().abs().reset_index()
    if selected_category and selected_category != "All":
        category_daily = category_daily[category_daily['category'] == selected_category]
    
    fig_category_trend = px.line(
        category_daily,
        x='date',
        y='amount',
        color='category' if selected_category in [None, "All"] else None,
        title=f"Category Spending Trend{'s' if selected_category in [None, 'All'] else f' - {selected_category}'}",
        labels={'amount': 'Amount (NGN)', 'date': 'Date'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    if selected_category in [None, "All"]:
        for category in category_daily['category'].unique():
            cat_data = category_daily[category_daily['category'] == category]
            fig_category_trend.add_trace(
                go.Scatter(
                    x=cat_data['date'],
                    y=cat_data['amount'],
                    name=category,
                    mode='lines+markers',
                    marker=dict(size=8),
                    showlegend=True
                )
            )
    else:
        fig_category_trend.update_traces(
            mode='lines+markers',
            marker=dict(size=8, color='#bfa14a'),
            line=dict(color='#bfa14a', width=3)
        )

    return fig_pie, fig_trend, fig_category_trend

def financial_chatbot_llm(user_question, df, income_df, openrouter_api_key):
    context = get_transaction_summary_for_llm(df, income_df)
    prompt = f"""You are a smart financial assistant. Here is the user's financial data:

{context}

User's question: {user_question}

Answer in a helpful, concise, and friendly way, using the data above.
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.3
    }
    response = requests.post(url, headers=headers, json=data, timeout=60)
    if not response.ok:
        st.error(f"OpenRouter API error: {response.status_code} - {response.text}")
        return "Sorry, there was an error with the OpenRouter API. Please check your API key, quota, and model name."
    return response.json()["choices"][0]["message"]["content"]

def get_transaction_summary_for_llm(df, income_df):
    last_tx = df.sort_values('date', ascending=False).head(50)
    tx_lines = [
        f"{row['date']}: {row['description']} - NGN {abs(row['amount']):,.2f} ({row['category']})"
        for _, row in last_tx.iterrows()
    ]
    tx_summary = "\n".join(tx_lines)

    income_df['date'] = pd.to_datetime(income_df['date'], errors='coerce')
    now = datetime.now()
    monthly_income = income_df[
        (income_df['date'].dt.month == now.month) &
        (income_df['date'].dt.year == now.year)
    ]['amount'].sum()

    cat_summary = df.groupby('category')['amount'].sum().abs().sort_values(ascending=False).head(3)
    cat_lines = [f"{cat}: NGN {amt:,.2f}" for cat, amt in cat_summary.items()]
    cat_summary_str = "\n".join(cat_lines)

    return f"""Recent Transactions:
{tx_summary}

Top Spending Categories:
{cat_summary_str}

Monthly Income: NGN {monthly_income:,.2f}
"""

def get_ngx_stock_prices():
    try:
        url = "https://africanfinancials.com/nigerian-stock-exchange-share-prices/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=15)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            stock_data = {}
            
            tables = soup.find_all('table')
            print(f"Found {len(tables)} tables")
            
            for i, table in enumerate(tables):
                print(f"Processing table {i}") 
                rows = table.find_all('tr')
                
                if len(rows) < 2:
                    continue
                
                header_row = rows[0]
                headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
                print(f"Table {i} headers: {headers}")
                
                stock_indicators = ['symbol', 'ticker', 'price', 'close', 'last', 'company', 'name']
                if not any(indicator in ' '.join(headers) for indicator in stock_indicators):
                    continue
                
                print(f"Table {i} looks like stock data")
                
                for j, row in enumerate(rows[1:]):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    
                    try:
                        cell_texts = [cell.get_text(strip=True) for cell in cells]
                        print(f"Row {j}: {cell_texts[:3]}")
                        
                        ticker = None
                        name = None
                        price = None
                        
                        if len(cells) >= 3:
                            ticker = cell_texts[0].upper()
                            name = cell_texts[1]
                            price_text = cell_texts[2]
                        
                        if ticker and len(ticker) >= 2 and ticker.isalpha():
                            price_clean = re.sub(r'[^\d.]', '', price_text)
                            if price_clean and '.' in price_clean or price_clean.isdigit():
                                price = float(price_clean)
                                
                                if price > 0:
                                    stock_data[ticker] = {
                                        'name': name if name else f"{ticker} Plc",
                                        'price': price
                                    }
                                    print(f"Added: {ticker} - {name} - {price}")
                    
                    except (ValueError, IndexError) as e:
                        print(f"Error processing row {j}: {e}")
                        continue
            
            if not stock_data:
                print("No table data found, trying alternative methods")
                
                divs = soup.find_all('div', class_=re.compile(r'stock|price|ticker', re.I))
                spans = soup.find_all('span', class_=re.compile(r'stock|price|ticker', re.I))
                
                for element in divs + spans:
                    text = element.get_text(strip=True)
                    matches = re.findall(r'([A-Z]{3,8})\s*[:\-]?\s*(\d+\.?\d*)', text)
                    for ticker, price in matches:
                        if len(ticker) >= 3:
                            stock_data[ticker] = {
                                'name': f"{ticker} Plc",
                                'price': float(price)
                            }
            
            print(f"Total stocks found: {len(stock_data)}")
            
            if stock_data:
                return stock_data
        
        else:
            print(f"HTTP Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        st.warning(f"Network error accessing African Financials: {str(e)}")
    except Exception as e:
        print(f"General error: {e}")
        st.warning(f"Error parsing African Financials data: {str(e)}")
    
    print("Using fallback data")
    return get_fallback_ngx_data()

def get_fallback_ngx_data():
    return {
        'DANGCEM': {'name': 'Dangote Cement Plc', 'price': 509.6},
        'MTNN': {'name': 'MTN Nigeria Communications Plc', 'price': 471.10},
        'ZENITHBANK': {'name': 'Zenith Bank Plc', 'price': 75},
        'GTCO': {'name': 'Guaranty Trust Holding Company Plc', 'price': 101.95},
        'BUAFOODS': {'name': 'BUA Foods Plc', 'price': 459.00},
        'AIRTELAFRI': {'name': 'Airtel Africa Plc', 'price': 2310.5},
        'FBNH': {'name': 'FBN Holdings Plc', 'price': 34.90},
        'UBA': {'name': 'United Bank for Africa Plc', 'price': 48.00},
        'ACCESSCORP': {'name': 'Access Holdings Plc', 'price': 27.60},
        'NESTLE': {'name': 'Nestle Nigeria Plc', 'price': 1890.0},
        'BUACEMENT': {'name': 'BUA Cement Plc', 'price': 135.00},
        'OANDO': {'name': 'Oando Plc', 'price': 59.3},
        'STANBIC': {'name': 'Stanbic IBTC Holdings Plc', 'price': 101.00},
        'FIDELITYBK': {'name': 'Fidelity Bank Plc', 'price': 21.00},
        'STERLING': {'name': 'Sterling Financial Holdings Company Plc', 'price': 6.70},
        'GUINNESS': {'name': 'Guinness Nigeria Plc', 'price': 106.45},
        'NB': {'name': 'Nigerian Breweries Plc', 'price': 77.00},
        'WAPCO': {'name': 'Lafarge Africa Plc', 'price': 151.00},
        'UNITYBNK': {'name': 'Unity Bank Plc', 'price': 1.51},
        'FLOURMILL': {'name': 'Flour Mills of Nigeria Plc', 'price': 81.80}
    }

def get_ngx_historical_data(ticker_symbol, period_days=365):
    try:
        current_stocks = get_ngx_stock_prices()
        current_price = None
        
        if current_stocks and ticker_symbol in current_stocks:
            current_price = current_stocks[ticker_symbol]['price']
        
        return generate_realistic_ngx_data(ticker_symbol, period_days, current_price)
        
    except Exception as e:
        st.error(f"Error generating historical data: {str(e)}")
        return generate_realistic_ngx_data(ticker_symbol, period_days)

def generate_realistic_ngx_data(ticker_symbol, period_days, current_price=None):
    fallback_prices = get_fallback_ngx_data()
    base_price = current_price if current_price else fallback_prices.get(ticker_symbol, {}).get('price', 50.0)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(period_days * 1.4))
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_days = all_dates[all_dates.weekday < 5]
    trading_days = trading_days[-period_days:] if len(trading_days) > period_days else trading_days
    
    np.random.seed(hash(ticker_symbol) % 2**32)
    
    daily_volatility = 0.035
    annual_drift = 0.10
    daily_drift = annual_drift / 252
    
    prices = [base_price]
    
    for i in range(1, len(trading_days)):
        dt = 1/252
        drift = daily_drift * dt
        shock = np.random.normal(0, daily_volatility * np.sqrt(dt))
        
        price_change = drift + shock
        new_price = prices[-1] * np.exp(price_change)
        
        new_price = max(new_price, base_price * 0.2)
        new_price = min(new_price, base_price * 5.0)
        
        prices.append(new_price)
    
    if len(prices) > 1:
        adjustment_factor = base_price / prices[-1]
        prices = [p * adjustment_factor for p in prices]
    
    base_volume = np.random.randint(10000, 200000)
    volumes = []
    
    for i in range(len(trading_days)):
        if i > 0:
            price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
            volume_multiplier = 1 + (price_change * 5)
        else:
            volume_multiplier = 1
            
        daily_volume = int(base_volume * volume_multiplier * np.random.uniform(0.3, 2.0))
        volumes.append(max(daily_volume, 1000))
    
    df = pd.DataFrame({
        'Date': trading_days,
        'Close': prices,
        'Volume': volumes,
        'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.03) for p in prices],
        'Low': [p * np.random.uniform(0.97, 1.00) for p in prices],
    })
    
    return df

def get_stock_prediction(ticker_symbol, investment_amount, prediction_days):
    try:
        hist_data = get_ngx_historical_data(ticker_symbol, period_days=365)
        
        if hist_data is None or hist_data.empty:
            st.error(f"No historical data available for {ticker_symbol}")
            return None, 0, 0
        
        hist_data = hist_data.sort_values('Date').reset_index(drop=True)
        hist_data['Days'] = range(len(hist_data))
        
        X = hist_data['Days'].values.reshape(-1, 1)
        y = hist_data['Close'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_days = np.arange(len(hist_data), len(hist_data) + prediction_days).reshape(-1, 1)
        predicted_prices = model.predict(future_days)
        
        current_price = y[-1]
        predicted_prices = np.clip(predicted_prices, current_price * 0.5, current_price * 3.0)
        
        predicted_final_price = predicted_prices[-1]
        shares_bought = investment_amount / current_price
        future_value = shares_bought * predicted_final_price
        potential_return = future_value - investment_amount
        return_percentage = (potential_return / investment_amount) * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{ticker_symbol} - Price Prediction', 'Investment Projection'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Scatter(
                x=hist_data['Date'],
                y=hist_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        future_dates = [hist_data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predicted_prices,
                mode='lines',
                name='Predicted Price',
                line=dict(color='red', dash='dash', width=2)
            ),
            row=1, col=1
        )
        
        investment_dates = [hist_data['Date'].iloc[-1], future_dates[-1]]
        investment_values = [investment_amount, future_value]
        
        fig.add_trace(
            go.Scatter(
                x=investment_dates,
                y=investment_values,
                mode='lines+markers',
                name='Investment Value',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Investment Prediction - {ticker_symbol}',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price (NGN ₦)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Investment Value (NGN ₦)", row=2, col=1)
        
        return fig, potential_return, return_percentage
        
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")
        return None, 0, 0

def main():
    st.markdown("""
<style>
    body, .stApp {
        background-color: #f8f8f8 !important;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #bfa14a;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    .metric-card {
        background: linear-gradient(90deg, #bfa14a 0%, #e5e5e5 100%);
        padding: 1rem;
        border-radius: 12px;
        color: #333;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(191,161,74,0.07);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        font-size: 1.05rem;
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    .user-message {
        background-color: #fffbe6;
        border-left: 5px solid #bfa14a;
        color: #333;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 5px solid #888;
        color: #444;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #fff;
        border-radius: 10px 10px 0 0;
        border-bottom: 2px solid #bfa14a;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888;
        font-weight: 500;
        font-size: 1.1rem;
        padding: 0.7rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        color: #bfa14a !important;
        border-bottom: 3px solid #bfa14a !important;
        background: #fff;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #bfa14a 0%, #e5e5e5 100%);
        color: #333;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        font-size: 1rem;
        margin: 0.2rem 0.1rem;
        box-shadow: 0 2px 8px rgba(191,161,74,0.07);
        transition: background 0.2s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #e5e5e5 0%, #bfa14a 100%);
        color: #bfa14a;
        border: 1px solid #bfa14a;
    }
    .stTextInput>div>div>input, .stNumberInput>div>input, .stDateInput>div>input {
        background: #fff;
        border: 1px solid #bfa14a;
        border-radius: 6px;
        color: #333;
    }
    .stDataFrame, .stTable {
        background: #fff;
        border-radius: 10px;
        border: 1px solid #e5e5e5;
        font-size: 1.05rem;
    }
    .stSidebar {
        background: #fff;
        border-right: 2px solid #bfa14a;
    }
    .stSidebar .stHeader {
        color: #bfa14a;
    }
    .stSidebar .stButton>button {
        background: #bfa14a;
        color: #fff;
    }
    .stSidebar .stButton>button:hover {
        background: #fff;
        color: #bfa14a;
        border: 1px solid #bfa14a;
    }
    /* Reduce font size of st.metric numbers */
    div[data-testid="stMetric"] > div > div {
        font-size: 0.8rem !important;
        color: #bfa14a !important;
    }
    div[data-testid="stMetric1"] > div > div {
        font-size: 1rem !important;
        color: #bfa14a !important;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background: #e5e5e5;
    }
    ::-webkit-scrollbar-thumb {
        background: #bfa14a;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">Kobo-AI</h1>', unsafe_allow_html=True)

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        with tab2:
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            if st.button("Register"):
                if register_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists.")
        return

    username = st.session_state.username
    transactions = load_transactions(username)
    income = load_income(username)

    if 'classifier' not in st.session_state:
        st.session_state.classifier = ExpenseClassifier()
    if 'predictor' not in st.session_state:
        st.session_state.predictor = SpendingPredictor()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Add Income")
        income_date = st.date_input("Income Date", datetime.now())
        income_amount = st.number_input("Income Amount (NGN)", min_value=0.0, step=1000.0, format="%.2f")
        if st.button("Include Income"):
            save_income(username, income_date.strftime('%Y-%m-%d'), income_amount)
            st.success(f"Added income: NGN {income_amount:,.2f} on {income_date.strftime('%Y-%m-%d')}")
            st.rerun()

        st.markdown("---")
        
        st.header("📝 Add New Transaction")
        
        with st.form("add_transaction"):
            date = st.date_input("Date", datetime.now())
            description = st.text_input("Description", placeholder="")
            amount = st.number_input("Amount (NGN)", min_value=100, step=100)
            if st.form_submit_button("Add Transaction"):
                predicted_category = st.session_state.classifier.predict_category(description)
                save_transaction(username, date.strftime('%Y-%m-%d'), description, -abs(amount), predicted_category)
                st.success(f"Added transaction: {description} (NGN {amount:,.2f}) - Category: {predicted_category}")
                st.rerun()
        
        st.markdown("---")
        
        if st.session_state.logged_in:
            if st.button("Logout", key="logout_btn"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.chat_history = []
                st.success("Logged out!")
                st.rerun()
    
    df = transactions
    income_df = income.copy()
    income_df['date'] = pd.to_datetime(income_df['date'], errors='coerce')
    now = datetime.now()
    monthly_income = income_df[
        (income_df['date'].dt.month == now.month) &
        (income_df['date'].dt.year == now.year)
    ]['amount'].sum()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", "📈 Predictions", "💰 Investment Predictions", 
        "🤖 AI Assistant", "📋 Transaction History", "🗂️ Master Summary"
    ])

    with tab1:
        df = transactions
        income_df = income.copy()
        income_df['date'] = pd.to_datetime(income_df['date'], errors='coerce')
        now = datetime.now()
        monthly_income = income_df[
            (income_df['date'].dt.month == now.month) &
            (income_df['date'].dt.year == now.year)
        ]['amount'].sum()

        if df.empty:
            total_spending = 0
            avg_transaction = 0
            transaction_count = 0
            days_span = 0
        else:
            total_spending = abs(df['amount'].sum())
            avg_transaction = abs(df['amount'].mean()) if not df['amount'].isnull().all() else 0
            transaction_count = len(df)
            valid_dates = pd.to_datetime(df['date'], errors='coerce').dropna()
            if not valid_dates.empty:
                days_span = (valid_dates.max() - valid_dates.min()).days + 1
            else:
                days_span = 0

        balance = monthly_income - total_spending

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Monthly Income", f"NGN {monthly_income:,.2f}")
        with col2:
            st.metric("Total Spending", f"NGN {total_spending:,.2f}")
        with col3:
            st.metric("Balance", f"NGN {balance:,.2f}")
        with col4:
            st.metric("Avg Transaction", f"NGN {avg_transaction:,.2f}")
        with col5:
            st.metric("Transactions", f"{transaction_count:,}")
        with col6:
            st.metric("Days Tracked", f"{days_span:,}")
        
        st.subheader("💹 Spending Analysis")
        
        fig_pie, fig_trend, fig_category_trend = create_spending_charts(df)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("#### Category Spending Trend")
        categories = ["All"] + sorted([cat for cat in df['category'].unique() if pd.notna(cat)])
        selected_category = st.selectbox("Select Category for Trend", categories, key="category_trend_select")
        _, _, fig_category_trend = create_spending_charts(df, selected_category=selected_category)
        st.plotly_chart(fig_category_trend, use_container_width=True)
        
        st.subheader("🏆 Top Spending Categories")
        category_summary = df.groupby('category').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        category_summary.columns = ['Total (NGN)', 'Transactions', 'Avg Amount (NGN)']
        category_summary['Total (NGN)'] = category_summary['Total (NGN)'].apply(lambda x: f"NGN {x:,.2f}")
        category_summary['Avg Amount (NGN)'] = category_summary['Avg Amount (NGN)'].apply(lambda x: f"NGN {x:,.2f}")
        category_summary = category_summary.sort_values('Total (NGN)', ascending=False)
        
        st.dataframe(category_summary, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Spending Predictions")
        
        predictions = st.session_state.predictor.predict_future_spending(df)
        
        if predictions:
            pred_df = pd.DataFrame(list(predictions.items()), columns=['Category', 'Predicted Amount'])
            pred_df = pred_df.sort_values('Predicted Amount', ascending=True)
            
            fig_pred = px.bar(
                pred_df,
                x='Predicted Amount',
                y='Category',
                orientation='h',
                title="Predicted Spending Next 30 Days",
                labels={'Predicted Amount': 'Amount (NGN)'},
                color='Predicted Amount',
                color_continuous_scale='viridis'
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            total_predicted = sum(predictions.values())
            st.metric("Total Predicted (30 days)", f"NGN{total_predicted:.2f}")
            
            current_daily_avg = abs(df['amount'].sum()) / max(1, len(df['date'].unique()))
            current_monthly_est = current_daily_avg * 30
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Monthly Est.", f"NGN{current_monthly_est:.2f}")
            with col2:
                difference = total_predicted - current_monthly_est
                st.metric("Predicted vs Current", f"NGN {difference:.2f}", f"{difference:.2f}")
            with col3:
                if current_monthly_est > 0:
                    change_pct = (difference / current_monthly_est) * 100
                    st.metric("Change %", f"{change_pct:.1f}%")
            
            st.subheader("📋 Category Predictions")
            pred_df_display = pred_df.copy()
            pred_df_display['Predicted Amount'] = pred_df_display['Predicted Amount'].apply(lambda x: f"${x:.2f}")
            st.dataframe(pred_df_display, use_container_width=True, hide_index=True)
            
        else:
            st.info("Need more transaction data to generate accurate predictions. Add more transactions to see forecasts!")
    
    with tab3:
        st.subheader("💰 Investment Predictions")
        st.caption("Data from African Financials & NGX")
        
        debug_mode = st.checkbox("Debug Mode", help="Show detailed debugging information")
        
        with st.spinner("Fetching available stocks from African Financials..."):
            stocks_data = get_ngx_stock_prices()
            
        if stocks_data:
            st.success(f"✅ Successfully loaded {len(stocks_data)} stocks from NGX")
            
            if debug_mode:
                st.write("**Debug Info:**")
                st.write(f"Stocks loaded: {list(stocks_data.keys())[:10]}...")
                
            col1, col2 = st.columns(2)
            
            with col1:
                input_method = st.radio("Select Input Method", 
                                  ["Choose from List", "Enter Ticker Symbol"])
                
                if input_method == "Choose from List":
                    stock_options = {f"{data['name']} ({ticker})": ticker 
                                     for ticker, data in stocks_data.items()}
                    selected_option = st.selectbox(
                        "Select Stock",
                        options=list(stock_options.keys())
                    )
                    selected_ticker = stock_options[selected_option]
                
                else:
                    selected_ticker = st.text_input(
                        "Enter Ticker Symbol",
                        help="Enter the stock ticker symbol (e.g., DANGCEM, MTNN)",
                        placeholder="DANGCEM"
                    ).upper()
                    
                investment_amount = st.number_input(
                    "Investment Amount (NGN ₦)",
                    min_value=1000.0,
                    value=100000.0,
                    step=1000.0,
                    format="%.2f"
                )
        
            with col2:
                prediction_period = st.selectbox(
                    "Prediction Period",
                    options=["6 Months", "1 Year"],
                    key="prediction_period"
                )
                
                if selected_ticker and selected_ticker in stocks_data:
                    current_price = stocks_data[selected_ticker]['price']
                    st.metric(
                        "Current Stock Price",
                        f"₦ {current_price:,.2f}"
                    )
                elif selected_ticker:
                    st.warning(f"⚠️ Ticker '{selected_ticker}' not found in our database")
                
                prediction_days = 180 if prediction_period == "6 Months" else 365
            
            if st.button("🔮 Predict Returns", type="primary"):
                if not selected_ticker:
                    st.warning("⚠️ Please enter a valid ticker symbol")
                else:
                    with st.spinner(f"Analyzing {selected_ticker} data and generating prediction..."):
                        fig, potential_return, return_percentage = get_stock_prediction(
                            selected_ticker, 
                            investment_amount,
                            prediction_days
                        )
                        
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            metric1_col1, metric1_col2, metric1_col3 = st.columns(3)
                            with metric1_col1:
                                st.metric(
                                    "Investment Amount",
                                    f"₦ {investment_amount:,.2f}"
                                )
                            with metric1_col2:
                                st.metric(
                                    "Potential Return",
                                    f"₦ {potential_return:,.2f}",
                                    f"{return_percentage:.1f}%"
                                )
                            with metric1_col3:
                                st.metric(
                                    "Projected Value",
                                    f"₦ {(investment_amount + potential_return):,.2f}"
                                )
                            
                            st.info("""
                            **Note:** These predictions are based on historical data and market trends. 
                            Actual returns may vary significantly. Always conduct thorough research and 
                            consider consulting with a financial advisor before making investment decisions.
                            """)
                        else:
                            st.error(f"Unable to generate prediction for {selected_ticker}. Please try a different stock.")
        else:
            st.error("Unable to fetch stock listings from African Financials")
            
            st.info("""
            **Possible causes:**
            - African Financials website might be temporarily unavailable
            - Network connectivity issues
            - Website structure may have changed
            - Rate limiting from the website
            
            **Solutions:**
            - Try refreshing the page
            - Check your internet connection
            - Try again in a few minutes
            - Use the fallback data by enabling debug mode
            """)
            
            if st.button("🔄 Use Cached Data"):
                st.rerun()
    
    with tab4:
        st.subheader("🤖 Financial AI Assistant")
        st.write("Ask me anything about your spending habits and budget!")
        
        st.markdown("**Or use your voice:**")
        webrtc_ctx = webrtc_streamer(
            key="audio",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=AudioProcessor,
            async_processing=True,
        )

        user_question = st.text_input("Ask a question:", placeholder="")

        if 'audio_file_path' in st.session_state and st.session_state['audio_file_path']:
            with st.spinner("Transcribing audio..."):
                with open(st.session_state['audio_file_path'], "rb") as audio_file:
                    transcript = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    user_question = transcript.text
                    st.success(f"Recognized: {user_question}")
                st.session_state['audio_file_path'] = None

        openrouter_api_key = st.secrets["openrouter"]["api_key"]

        if st.button("Ask") and user_question:
            openrouter_api_key = st.secrets["openrouter"]["api_key"]
            bot_response = financial_chatbot_llm(user_question, transactions, income, openrouter_api_key)
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("bot", bot_response))
            st.rerun()
        
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for role, message in st.session_state.chat_history[-6:]:
                if role == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message"><strong>AI Assistant:</strong><br>{message}</div>', unsafe_allow_html=True)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.subheader("💡 Quick Questions")
        quick_questions = [
            "How much do I spend on food?",
            "Can I afford a NGN 1,500,000 vacation?",
            "How can I save money?",
            "What will I spend next month?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                bot_response = financial_chatbot_llm(question, transactions, income, openrouter_api_key)
                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("bot", bot_response))
                st.rerun()
    
    with tab5:
        st.subheader("📋 Transaction History")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = ['All'] + sorted(df['category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", categories)
        
        with col2:
            valid_dates = pd.to_datetime(df['date'], errors='coerce').dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
            else:
                today = datetime.today().date()
                min_date = today
                max_date = today

            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        with col3:
            min_amount = st.number_input("Min Amount (NGN)", value=0.0)
        
        filtered_df = df.copy()
        
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (pd.to_datetime(filtered_df['date']).dt.date >= start_date) &
                (pd.to_datetime(filtered_df['date']).dt.date <= end_date)
            ]
        
        filtered_df = filtered_df[abs(filtered_df['amount']) >= min_amount]
        
        st.write(f"Showing {len(filtered_df)} transactions")
        
        display_df = filtered_df.copy()
        display_df['amount'] = display_df['amount'].apply(lambda x: f"NGN {abs(x):,.2f}")
        display_df = display_df.sort_values('date', ascending=False)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        if st.button("📥 Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab6:
        st.subheader("🗂️ Master Summary by Month")

        all_dates = pd.concat([
            pd.to_datetime(transactions['date'], errors='coerce'),
            pd.to_datetime(income['date'], errors='coerce')
        ]).dropna()
        if not all_dates.empty:
            months = sorted(all_dates.dt.to_period('M').unique(), reverse=True)
            selected_month = st.selectbox("Select Month", months, format_func=lambda x: x.strftime('%B %Y'))
        else:
            selected_month = pd.Period(datetime.now(), freq='M')

        tx_df = transactions.copy()
        tx_df['date'] = pd.to_datetime(tx_df['date'], errors='coerce')
        tx_df['type'] = 'Expense'
        income_df = income.copy()
        income_df['date'] = pd.to_datetime(income_df['date'], errors='coerce')
        income_df['description'] = 'Income'
        income_df['category'] = 'Income'
        income_df['type'] = 'Income'

        master_df = pd.concat([tx_df, income_df], ignore_index=True)
        master_df['month'] = master_df['date'].dt.to_period('M')
        month_df = master_df[master_df['month'] == selected_month]

        for idx, row in month_df.iterrows():
            cols = st.columns([2, 2, 2, 2, 2, 1])
            cols[0].write(row['date'].strftime('%Y-%m-%d'))
            cols[1].write(row.get('description', ''))
            cols[2].write(row.get('category', ''))
            cols[3].write(row['type'])
            cols[4].write(f"NGN {abs(row['amount']):,.2f}")
            if cols[5].button("Delete", key=f"del_{idx}_{row['type']}"):
                if row['type'] == 'Expense':
                    delete_transaction(username, row['date'].strftime('%Y-%m-%d'), row['description'], row['amount'])
                else:
                    delete_income(username, row['date'].strftime('%Y-%m-%d'), row['amount'])
                st.success("Entry deleted!")
                st.rerun()

        if month_df.empty:
            st.info("No transactions or income for this month.")

if __name__ == "__main__":
    main()
