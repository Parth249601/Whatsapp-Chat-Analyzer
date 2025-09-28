# helper.py

from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import emoji
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import networkx as nx
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import seaborn as sns
from transformers import pipeline
import streamlit as st # Import Streamlit to use its caching
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired
from dotenv import load_dotenv

#------------------AI CHATBOT------------------#

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

#----------------------------------------------#
# This code runs once when the module is imported, ensuring data is ready.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Initialize libraries ---
# Now this will succeed because the lexicon is guaranteed to be downloaded.
sentiments = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

# --- Standard Analysis Functions ---

@st.cache_data
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = [word for message in df['message'] for word in message.split()]
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = [link for message in df['message'] for link in re.findall(r'(https?://\S+)', message)]
    return num_messages, len(words), num_media_messages, len(links)

@st.cache_data
def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    # Make sure 'stop_hinglish.txt' is in the same directory
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().split()
    words = [word for message in temp['message'] for word in message.lower().split() if word not in stop_words]
    return pd.DataFrame(Counter(words).most_common(20))

@st.cache_data
def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = [c for message in df['message'] for c in message if c in emoji.EMOJI_DATA]
    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

@st.cache_data
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline

@st.cache_data
def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

@st.cache_data
def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

@st.cache_data
def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

@st.cache_data
def sentiment_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df.copy()
    # Check if sentiment score already exists to avoid re-calculation within the cached function
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = df['message'].apply(lambda x: sentiments.polarity_scores(x)['compound'])
    timeline = df.groupby(['year', 'month_num', 'month'])['sentiment_score'].mean().reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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

@st.cache_data
def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().split()

    def remove_stopwords(message):
        words = [word for word in message.lower().split() if word not in stop_words]
        return " ".join(words)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    # Use a copy to avoid SettingWithCopyWarning
    temp_copy = temp.copy()
    temp_copy['message'] = temp_copy['message'].apply(remove_stopwords)
    df_wc = wc.generate(temp_copy['message'].str.cat(sep=" "))
    return df_wc

@st.cache_data
def user_sentiment_analysis(df):
    # Make a copy to ensure the original cached df is not modified
    df_copy = df.copy()
    if 'sentiment_score' not in df_copy.columns:
        df_copy['sentiment_score'] = df_copy['message'].apply(lambda x: sentiments.polarity_scores(x)['compound'])
    
    user_df = df_copy[df_copy['user'] != 'group_notification']
    sentiment_df = user_df.groupby('user')['sentiment_score'].mean().sort_values(ascending=False).reset_index()
    return sentiment_df

@st.cache_data
def find_topics(df, num_topics=5, num_words=5):
    """Find conversation topics using LDA with improved preprocessing."""
    text_data = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]['message']
    
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().split())
        stop_words.update(['https', 'http', 'www'])

    processed_texts = []
    for doc in nlp.pipe(text_data, disable=["parser", "ner"]):
        tokens = [
            token.lemma_.lower().strip() 
            for token in doc 
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
               token.text.lower() not in stop_words and 
               not token.is_punct and 
               len(token.text) > 2
        ]
        if tokens:
            processed_texts.append(" ".join(tokens))

    if not processed_texts:
        return {"Error": ["Not enough text to model topics."]}

    vectorizer = CountVectorizer(ngram_range=(1, 2), max_df=0.90, min_df=3, stop_words=list(stop_words))
    X = vectorizer.fit_transform(processed_texts)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    topics = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-num_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f"Topic {topic_idx+1}"] = top_words
        
    return topics

@st.cache_data
@st.cache_data
def find_topics_bertopic(_df):
    """
    Finds topics using an optimized BERTopic model and returns cached results.
    """
    df = _df.copy()
    text_data = df[(df['user'] != 'group_notification') & (~df['message'].str.contains('<Media omitted>'))]['message'].tolist()

    if len(text_data) < 20: # Increased minimum size for better results
        return pd.DataFrame(), None, None

    # 1. Use a much lighter and faster embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Reduce UMAP components for speed
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    # 3. Increase min_cluster_size to get more significant topics
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    # 4. Use stopwords to remove noise and create better topic representations
    vectorizer_model = CountVectorizer(stop_words="english")
    
    # 5. Use KeyBERT to create more meaningful topic names
    representation_model = KeyBERTInspired()

    try:
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            language="english",
            nr_topics=10,  # Aim for the top 10 topics after initial creation
            verbose=False
        )
        topics, _ = topic_model.fit_transform(text_data)
        topic_info = topic_model.get_topic_info()
        
        return topic_info, topic_model, topics

    except Exception as e:
        print(f"BERTopic error: {e}")
        return pd.DataFrame(), None, None

# Visualization functions should typically not be cached, but the data they use should be.
def visualize_topics_over_time(df, topic_model, topics):
    """
    Generates an interactive plot of topic frequency over time.
    """
    filtered_df = df[(df['user'] != 'group_notification') & (~df['message'].str.contains('<Media omitted>'))].copy()

    if len(topics) != len(filtered_df):
        st.warning("Mismatch between topic count and document count. Cannot generate timeline.")
        return None

    docs = filtered_df['message'].tolist()
    timestamps = filtered_df['date'].tolist()
    
    topics_over_time_df = topic_model.topics_over_time(docs=docs, timestamps=timestamps, topics=topics, nr_bins=20)
    fig = topic_model.visualize_topics_over_time(topics_over_time_df)
    
    return fig

@st.cache_resource
def load_summarizer():
    """
    Loads a smaller, distilled summarization model.
    This model is smaller than t5-small and is specifically fine-tuned for summarization.
    """
    model = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    return model

# The summary generation itself should NOT be cached, as it depends on the date range, which changes frequently.
# The expensive part (loading the model) IS cached with @st.cache_resource.
def generate_summary(df, start_date, end_date, progress_bar):

    """
    More memory-efficiently generates a summary by processing messages in chunks
    without creating a single massive text string.
    """
    print("Generating summary...")
    summarizer = load_summarizer()

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    df_copy['date_only'] = df_copy['date'].dt.date
    mask = (df_copy['date_only'] >= start_date) & (df_copy['date_only'] <= end_date)
    filtered_df = df_copy.loc[mask]

    if filtered_df.empty:
        progress_bar.progress(1.0)
        return "No messages found in the selected date range to summarize."

    max_chunk_word_count = 400
    chunks = []
    current_chunk_text = ""
    
    for index, row in filtered_df.iterrows():
        message_text = f"{row['user']}: {row['message']}\n"
        
        if len(current_chunk_text.split()) + len(message_text.split()) > max_chunk_word_count:
            if current_chunk_text:
                chunks.append(current_chunk_text)
            current_chunk_text = message_text
        else:
            current_chunk_text += message_text

    if current_chunk_text:
        chunks.append(current_chunk_text)

    final_summary = ""
    total_chunks = len(chunks)
    
    print(f"Total chunks to summarize: {total_chunks}")
    if total_chunks == 0:
        progress_bar.progress(1.0)
        return "Not enough text in the selected date range to generate a summary."

    for i, chunk in enumerate(chunks):
        min_len = 30
        max_len = max(min_len + 10, int(len(chunk.split()) * 0.3))
        
        try:
            summary_result = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            final_summary += summary_result[0]['summary_text'] + " "
        except Exception as e:
            print(f"Could not summarize chunk {i+1}: {e}")
            
        progress_bar.progress((i + 1) / total_chunks)
        
    return final_summary.strip()

#------------------AI CHATBOT------------------#

load_dotenv()

# We try to get the key from Streamlit's secrets first, then from the local .env file
# This makes the code work both in deployment and locally.
try:
    # For Streamlit Community Cloud
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    # For local development
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# If the key is not found in either place, raise an error
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please set it in your Streamlit secrets or .env file.")
@st.cache_resource
def create_conversational_chain(_df):
    """
    Creates and caches the RAG conversational chain.
    """
    # 1. Combine all messages into a single text block
    full_text = "\n".join(
        _df.apply(lambda row: f"{row['user']}: {row['message']}", axis=1)
    )

    # 2. Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(full_text)

    # 3. Create embeddings using a local model (no API key needed here)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # 4. Create the LLM and the Prompt Template
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context just say, "answer is not available in the context". Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # 5. Initialize the model with the securely loaded API key
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        google_api_key=GOOGLE_API_KEY,  # Use the secure key
        temperature=0.3, 
        convert_system_message_to_human=True
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return vector_store, chain
#----------------------------------------------#