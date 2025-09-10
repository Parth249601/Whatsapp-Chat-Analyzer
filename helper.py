from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import networkx as nx
import re
import google.generativeai as genai

# Initialize libraries
sentiments = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

# --- Standard Analysis Functions ---

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = [word for message in df['message'] for word in message.split()]
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = [link for message in df['message'] for link in re.findall(r'(https?://\S+)', message)]
    return num_messages, len(words), num_media_messages, len(links)

def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().split()
    words = [word for message in temp['message'] for word in message.lower().split() if word not in stop_words]
    return pd.DataFrame(Counter(words).most_common(20))

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = [c for message in df['message'] for c in message if c in emoji.EMOJI_DATA]
    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

def sentiment_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df.copy()
    df['sentiment_score'] = df['message'].apply(lambda x: sentiments.polarity_scores(x)['compound'])
    timeline = df.groupby(['year', 'month_num', 'month'])['sentiment_score'].mean().reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline

def topic_modeling(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    stop_words = set(stopwords.words('english'))
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        custom_stop_words = f.read().splitlines()
    stop_words.update(custom_stop_words)
    def clean_text(text):
        tokens = word_tokenize(text.lower())
        return [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
    processed_docs = temp['message'].apply(clean_text)
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    if not corpus or not dictionary: return []
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    return lda_model.print_topics(num_words=5)

def named_entity_recognition(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    full_text = ' '.join(df['message'])
    doc = nlp(full_text)
    entities = {'PERSON': [], 'GPE': [], 'ORG': []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    top_persons = pd.DataFrame(Counter(entities['PERSON']).most_common(10), columns=['Person', 'Count'])
    top_places = pd.DataFrame(Counter(entities['GPE']).most_common(10), columns=['Place (GPE)', 'Count'])
    top_orgs = pd.DataFrame(Counter(entities['ORG']).most_common(10), columns=['Organization', 'Count'])
    return top_persons, top_places, top_orgs

def create_interaction_network(selected_user, df):
    if selected_user != 'Overall':
        return None, None
    users_to_exclude = ['Meta AI']
    df_filtered = df[(df['user'] != 'group_notification') & (~df['user'].isin(users_to_exclude))]
    user_counts = df_filtered['user'].value_counts().to_dict()
    G = nx.DiGraph()
    for user in df_filtered['user'].unique():
        G.add_node(user)
    for i in range(1, len(df_filtered)):
        sender, receiver = df_filtered['user'].iloc[i-1], df_filtered['user'].iloc[i]
        if sender != receiver:
            if G.has_edge(sender, receiver):
                G[sender][receiver]['weight'] += 1
            else:
                G.add_edge(sender, receiver, weight=1)
    return G, user_counts

def calculate_response_times(selected_user, df):
    if selected_user != 'Overall':
        return pd.DataFrame(columns=['user', 'avg_response_time_minutes'])
    df_filtered = df[df['user'] != 'group_notification']
    response_data = []
    for i in range(1, len(df_filtered)):
        current_user, previous_user = df_filtered['user'].iloc[i], df_filtered['user'].iloc[i-1]
        if current_user != previous_user:
            time_diff = df_filtered['date'].iloc[i] - df_filtered['date'].iloc[i-1]
            diff_minutes = time_diff.total_seconds() / 60
            if diff_minutes < 1440:
                response_data.append({'user': current_user, 'response_time': diff_minutes})
    if not response_data:
        return pd.DataFrame(columns=['user', 'avg_response_time_minutes'])
    response_df = pd.DataFrame(response_data)
    avg_response_df = response_df.groupby('user')['response_time'].mean().reset_index()
    avg_response_df.rename(columns={'response_time': 'avg_response_time_minutes'}, inplace=True)
    return avg_response_df.sort_values(by='avg_response_time_minutes')

# --- AGENTIC AI TOOLS AND MAIN FUNCTION ---

# CORRECTED a tool to be more flexible
def most_busy_users(df, year: int = None, month: str = None):
    """Finds the most active users in the chat. Can be filtered by year and/or month."""
    df_filtered = df.copy()
    if year:
        df_filtered = df_filtered[df_filtered['year'] == year]
    if month:
        df_filtered = df_filtered[df_filtered['month'] == month.capitalize()]
    if df_filtered.empty:
        return "No messages found for the specified period.", ""
    x = df_filtered['user'].value_counts().head()
    df_percent = round((df_filtered['user'].value_counts() / df_filtered.shape[0]) * 100, 2).reset_index().rename(
        columns={'user': 'percent', 'index': 'name'})
    return x, df_percent

# NEW tool
def summarize_day(df, date_str: str):
    """Summarizes the conversation for a specific day."""
    try:
        target_date = pd.to_datetime(date_str).date()
        df_day = df[df['date'].dt.date == target_date]
        if df_day.empty:
            return "No messages were found for that day."
        conversation = "\n".join(df_day['user'] + ': ' + df_day['message'])
        # NOTE: API key is configured once in the main agentic function
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Summarize the following WhatsApp conversation in a few sentences:\n\n{conversation}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not summarize the day. Error: {e}"

# NEW tool
def get_sentiment_for_day(df, date_str: str):
    """Calculates the average sentiment for a specific day."""
    try:
        target_date = pd.to_datetime(date_str).date()
        df_day = df[df['date'].dt.date == target_date].copy()
        if df_day.empty:
            return "No messages were found for that day to analyze sentiment."
        df_day['sentiment_score'] = df_day['message'].apply(lambda x: sentiments.polarity_scores(x)['compound'])
        avg_sentiment = df_day['sentiment_score'].mean()
        mood = "Neutral"
        if avg_sentiment > 0.05: mood = "Positive"
        elif avg_sentiment < -0.05: mood = "Negative"
        return f"The average sentiment on {date_str} was {avg_sentiment:.2f}, which is considered {mood}."
    except Exception as e:
        return f"Could not get sentiment. Error: {e}"

# UPGRADED AGENT
def get_agentic_response(df, question):
    """Uses the Gemini model with a toolbox of functions to answer questions."""
    try:
        # IMPORTANT: Replace "YOUR_API_KEY" with your actual key
        genai.configure(api_key="AIzaSyDwCiYitGboPF-xCPUbUSbBLg7X9zSiEZU") 
    except Exception as e:
        return f"API Key configuration error: {e}"

    tools = {'most_busy_users': most_busy_users, 'summarize_day': summarize_day, 'get_sentiment_for_day': get_sentiment_for_day}
    
    function_declarations = [
        {"name": "most_busy_users", "description": "Finds the most active users in the chat. Call this without arguments for overall, or with year/month for a specific period.", "parameters": {"type": "object", "properties": {"year": {"type": "integer"}, "month": {"type": "string"}}}},
        {"name": "summarize_day", "description": "Summarizes the entire conversation for a single, specific day.", "parameters": {"type": "object", "properties": {"date_str": {"type": "string", "description": "The date in YYYY-MM-DD format, e.g., '2025-07-28'"}}, "required": ["date_str"]}},
        {"name": "get_sentiment_for_day", "description": "Gets the average sentiment or 'mood' for a single, specific day.", "parameters": {"type": "object", "properties": {"date_str": {"type": "string", "description": "The date in YYYY-MM-DD format, e.g., '2024-12-25'"}}, "required": ["date_str"]}}
    ]

    model = genai.GenerativeModel('gemini-pro', tools=function_declarations)
    chat = model.start_chat()

    try:
        response = chat.send_message(f"You are a helpful WhatsApp chat analyst. Here is a user's question: '{question}'. Use your available tools to answer it.")
        function_call = response.candidates[0].content.parts[0].function_call
    except (IndexError, AttributeError):
        return response.text
    except Exception as e:
        return f"Error during initial API call: {e}"

    function_name = function_call.name
    args = {key: value for key, value in function_call.args.items()}
    
    if function_name in tools:
        function_result = tools[function_name](df, **args)
        if isinstance(function_result, tuple):
            result_str = function_result[0].to_string()
        else:
            result_str = str(function_result)
        final_response = chat.send_message(f"The result of the tool call is: {result_str}. Now, provide a final, human-readable answer to the user's original question: '{question}'")
        return final_response.text
    else:
        return "The model suggested a function that is not available."