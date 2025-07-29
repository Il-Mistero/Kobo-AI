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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import openai
import av
import tempfile

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

def create_spending_charts(df):
    
    category_totals = df.groupby('category')['amount'].sum().abs()
    
    fig_pie = px.pie(
        values=category_totals.values,
        names=category_totals.index,
        title="Spending by Category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    df['date'] = pd.to_datetime(df['date'])
    daily_spending = df.groupby('date')['amount'].sum().abs().reset_index()
    
    fig_trend = px.line(
        daily_spending,
        x='date',
        y='amount',
        title="Daily Spending Trend",
        labels={'amount': 'Amount (NGN)', 'date': 'Date'}
    )
    fig_trend.update_traces(line_color='#1f77b4', line_width=3)
    
    category_daily = df.groupby(['date', 'category'])['amount'].sum().abs().reset_index()
    
    fig_category_trend = px.line(
        category_daily,
        x='date',
        y='amount',
        color='category',
        title="Category Spending Trends",
        labels={'amount': 'Amount (NGN)', 'date': 'Date'}
    )
    
    return fig_pie, fig_trend, fig_category_trend

def financial_chatbot(user_question, df, predictions, monthly_income):
    question_lower = user_question.lower()
    if df.empty:
        return "No transactions found. Please add some transactions first!"

    df['amount'] = df['amount'].fillna(0)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['category'] = df['category'].fillna('Other')

    total_spending = abs(df['amount'].sum())
    avg_daily = total_spending / max(1, df['date'].nunique())
    if df['category'].isnull().all():
        top_category = "Other"
        top_category_amount = 0
    else:
        top_category = df.groupby('category')['amount'].sum().abs().idxmax()
        top_category_amount = abs(df.groupby('category')['amount'].sum().get(top_category, 0))

    current_month = datetime.now().month
    current_year = datetime.now().year
    monthly_spending = abs(df[(df['date'].dt.month == current_month) & (df['date'].dt.year == current_year)]['amount'].sum())

    if any(word in question_lower for word in ['afford', 'budget', 'vacation', 'trip']):
        monthly_prediction = sum(predictions.values()) if predictions else monthly_spending
        available_budget = max(0, monthly_income - monthly_prediction)

        if 'vacation' in question_lower or 'trip' in question_lower:
            return f"""💰 **Here's my take:**

Based on your current spending patterns, your predicted monthly expenses are NGN{monthly_prediction:,.2f}.

With a monthly income of NGN{monthly_income:,.2f}, you might have around NGN{available_budget:,.2f} available for discretionary spending like vacations.

Your biggest expense category is {top_category} (NGN{top_category_amount:,.2f}), so look for savings there first!"""
    
    elif any(word in question_lower for word in ['save', 'saving', 'reduce', 'cut']):
        savings_tips = f"""💡 **Personalized Savings Tips:**

Based on your spending of NGN{total_spending:.2f} over {len(df)} transactions:

1. **{top_category}** is your largest expense (NGN{top_category_amount:.2f})
   - Try to reduce this by 10-15% to save NGN {top_category_amount * 0.125:.2f}

2. **Daily Average:** You spend NGN {avg_daily:.2f} per day

3. **Category-specific tips:**"""
        
        category_tips = {
            'Food & Dining': "- Cook more meals at home\n- Use meal planning apps\n- Limit eating out to 2-3 times per week",
            'Transportation': "- Use public transport or carpool with people going in the same direction",
            'Shopping': "- Wait 24 hours before non-essential purchases\n- Use shopping lists",
            'Entertainment': "- Try free activities like chatting with a neighbour or taking a walk in your estate"
        }
        
        if top_category in category_tips:
            savings_tips += f"\n{category_tips[top_category]}"
        
        return savings_tips
    
    elif any(word in question_lower for word in ['spend', 'spending', 'expense']):
        return f"""📊 **Your Spending Summary:**

- **Total Spending:** NGN {total_spending:.2f}
- **Average Daily:** NGN {avg_daily:.2f}
- **Top Category:** {top_category} (NGN {top_category_amount:.2f})
- **Number of Transactions:** {len(df)}

**Predicted Next Month:** NGN {sum(predictions.values()):.2f} if predictions else 'Need more data for predictions'"""
    
    elif any(word in question_lower for word in ['predict', 'future', 'forecast']):
        if predictions:
            pred_text = "🔮 **Spending Predictions for Next 30 Days:**\n\n"
            for category, amount in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                pred_text += f"- **{category}:** NGN {amount:.2f}\n"
            pred_text += f"\n**Total Predicted:** NGN {sum(predictions.values()):.2f}"
            return pred_text
        else:
            return "I need more transaction data to make accurate predictions. Please add more transactions!"
    
    else:
        return f"""🤖 **I can help you with:**

- **Budget questions:** "Can I afford a vacation?" 
- **Savings advice:** "How can I save money?"
- **Spending analysis:** "How much do I spend on food?"
- **Predictions:** "What will I spend next month?"

Your current spending summary: NGN {total_spending:.2f} total, NGN {avg_daily:.2f} daily average.
Top category: {top_category} (NGN{top_category_amount:.2f})"""

def main():
    st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    /* Reduce font size of st.metric numbers */
    div[data-testid="stMetric"] > div > div {
        font-size: 1.0rem !important;
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

    if 'transactions' not in st.session_state:
        st.session_state.transactions = pd.DataFrame(columns=['date', 'description', 'amount', 'category'])
    if 'income' not in st.session_state:
        st.session_state.income = pd.DataFrame(columns=['date', 'amount'])
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
            new_income = pd.DataFrame({
                'date': [income_date.strftime('%Y-%m-%d')],
                'amount': [income_amount]
            })
            st.session_state.income = pd.concat([st.session_state.income, new_income], ignore_index=True)
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
                
                new_transaction = pd.DataFrame({
                    'date': [date.strftime('%Y-%m-%d')],
                    'description': [description],
                    'amount': [-abs(amount)],
                    'category': [predicted_category]
                })
                
                st.session_state.transactions = pd.concat([
                    st.session_state.transactions, 
                    new_transaction
                ], ignore_index=True)
                
                st.success(f"Added transaction: {description} (NGN {amount:.2f}) - Category: {predicted_category}")
                st.rerun()
        
        st.markdown("---")
        
        st.header("🔧 Data Management")
        
        if st.button("Load Sample Data"):
            st.session_state.transactions = generate_sample_data()
            st.success("Sample data loaded!")
            st.rerun()
        
        if st.button("Train AI Classifier"):
            success = st.session_state.classifier.train_model(st.session_state.transactions)
            if success:
                st.success("AI classifier trained successfully!")
            else:
                st.warning("Need more data to train classifier")
        
    
    income_df = st.session_state.income.copy()
    income_df['date'] = pd.to_datetime(income_df['date'], errors='coerce')
    now = datetime.now()
    monthly_income = income_df[
        (income_df['date'].dt.month == now.month) &
        (income_df['date'].dt.year == now.year)
    ]['amount'].sum()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "📈 Predictions", "🤖 AI Assistant", "📋 Transaction History", "🗂️ Master Summary"
    ])

    with tab1:
        df = st.session_state.transactions

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
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_trend, use_container_width=True)
        
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

        if st.button("Ask") and user_question:
            predictions = st.session_state.predictor.predict_future_spending(df)
            bot_response = financial_chatbot(user_question, df, predictions, st.session_state.monthly_income)
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
                predictions = st.session_state.predictor.predict_future_spending(df)
                bot_response = financial_chatbot(question, df, predictions, st.session_state.monthly_income)
                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("bot", bot_response))
                st.rerun()
    
    with tab4:
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
    
    with tab5:
        st.subheader("🗂️ Master Summary by Month")

        all_dates = pd.concat([
            pd.to_datetime(st.session_state.transactions['date'], errors='coerce'),
            pd.to_datetime(st.session_state.income['date'], errors='coerce')
        ]).dropna()
        if not all_dates.empty:
            months = sorted(all_dates.dt.to_period('M').unique(), reverse=True)
            selected_month = st.selectbox("Select Month", months, format_func=lambda x: x.strftime('%B %Y'))
        else:
            selected_month = pd.Period(datetime.now(), freq='M')

        tx_df = st.session_state.transactions.copy()
        tx_df['date'] = pd.to_datetime(tx_df['date'], errors='coerce')
        tx_df['type'] = 'Expense'
        income_df = st.session_state.income.copy()
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
                    st.session_state.transactions = st.session_state.transactions.drop(idx)
                else:
                    st.session_state.income = st.session_state.income.drop(idx)
                st.success("Entry deleted!")
                st.rerun()

        if month_df.empty:
            st.info("No transactions or income for this month.")

if __name__ == "__main__":
    main()
