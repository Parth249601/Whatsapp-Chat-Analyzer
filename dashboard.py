# A minimal dashboard.py for debugging

import streamlit as st
import preprocessor
import pandas as pd # Add pandas import
import plotly.express as px
import helper

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Reads, decodes, and preprocesses the uploaded file."""
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    return df

# --- Streamlit UI Setup ---
st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat export file (.txt)")

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)

    if df.empty:
        st.error("Could not parse any messages from the chat file. Please ensure the format is correct.")
    else:
        # --- Sidebar Controls ---
        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Show analysis for:", user_list)

        if st.sidebar.button("Show Analysis"):
            # --- Main Analysis Area ---
            st.title(f"📊 Analysis for: {selected_user}")

            # Define the tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Overview", "🕒 Temporal Analysis", "📝 Text & Emoji Analysis", "🔬 Advanced Analysis", "📜 Conversation Summarizer"])

            # --- Tab 1: Overview ---
            with tab1:
                st.header("Top Statistics")
                num_messages, words, num_media, num_links = helper.fetch_stats(selected_user, df)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Messages", num_messages)
                with col2:
                    st.metric("Total Words", words)
                with col3:
                    st.metric("Media Shared", num_media)
                with col4:
                    st.metric("Links Shared", num_links)

                # Most Active Users (Only for Overall)
                if selected_user == 'Overall':
                    st.header("Most Active Users")
                    user_activity = df['user'].value_counts().head().reset_index()
                    user_activity.columns = ['User', 'Messages'] # Rename columns for clarity

                    # --- NEW Plotly Code ---
                    fig = px.bar(user_activity, x='User', y='Messages', color='User',
                                 title="Top 5 Most Active Users",
                                 labels={'Messages': 'Number of Messages'})
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Processed Data Snippet")
                st.dataframe(df.head())
            with tab2:
                st.header("Temporal Analysis")
                st.write("This is Tab 2.")

            with tab3:
                st.header("Text & Emoji Analysis")
                st.write("This is Tab 3.")

            with tab4:
                st.header("Advanced Analysis")
                st.write("This is Tab 4.")

            with tab5:
                st.header("Conversation Summarizer")
                # We'll use a simplified version of the date input to test
                min_date = df['date'].dt.date.min()
                max_date = df['date'].dt.date.max()
                st.date_input("Start date", value=min_date)
                st.write("This is the summarizer tab.")
                st.info("Change the date above. If the app doesn't freeze, the problem is in your original code.")

else:
    st.info("Awaiting for a WhatsApp chat file to be uploaded.")