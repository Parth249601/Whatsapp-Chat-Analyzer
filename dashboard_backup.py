import streamlit as st
import preprocessor
import helper
import plotly.express as px
import matplotlib.pyplot as plt # Still needed for WordCloud
import networkx as nx 
import seaborn as sns
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Reads, decodes, and preprocesses the uploaded file."""
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    return df

# --- Streamlit UI Setup ---
st.sidebar.title("WhatsApp Chat Analyzer")
st.sidebar.markdown("Analyze your WhatsApp chats to gain insights into your conversations.")

# File uploader
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

        # --- Initialize session state for analysis view ---
        if 'analysis_triggered' not in st.session_state:
            st.session_state.analysis_triggered = False

        if st.sidebar.button("Show Analysis"):
            st.session_state.analysis_triggered = True # Set the flag to True on click

        # --- Main Analysis Area: Show only if the button has been clicked ---
        if st.session_state.analysis_triggered:
            st.title(f"📊 Analysis for: {selected_user}")

            # Define the tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Temporal Analysis", "Text & Emoji Analysis", "Advanced Analysis", "Summarizer", "🤖 Chatbot Q&A"])

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


            # --- Tab 2: Temporal Analysis ---
            with tab2:
                st.header("Monthly Activity")
                timeline = helper.monthly_timeline(selected_user, df)
                if not timeline.empty:
                    fig = px.line(timeline, x='time', y='message', title='Monthly Message Count', markers=True)
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

                st.header("Daily & Weekly Activity")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Busiest Day of the Week")
                    busy_day = helper.week_activity_map(selected_user, df).reset_index()
                    busy_day.columns = ['Day', 'Messages']
                    
                    # --- NEW Plotly Code ---
                    fig = px.bar(busy_day, x='Day', y='Messages', color='Day', title="Most Messages by Day")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    st.subheader("Busiest Month")
                    busy_month = helper.month_activity_map(selected_user, df).reset_index()
                    busy_month.columns = ['Month', 'Messages']

                    # --- NEW Plotly Code ---
                    fig = px.bar(busy_month, x='Month', y='Messages', color='Month', title="Most Messages by Month")
                    st.plotly_chart(fig, use_container_width=True)

                st.header("Weekly Activity Heatmap")
                user_heatmap = helper.activity_heatmap(selected_user, df)
                if not user_heatmap.empty:
                    fig, ax = plt.subplots(figsize=(15, 8))
                    sns.heatmap(user_heatmap, cmap="YlGnBu", ax=ax)
                    st.pyplot(fig)

            # --- Tab 3: Text & Emoji Analysis ---
            with tab3:
                st.header("Most Common Words")
                most_common_df = helper.most_common_words(selected_user, df)
                if not most_common_df.empty:
                    fig, ax = plt.subplots()
                    sns.barplot(x=most_common_df[1], y=most_common_df[0], palette="plasma", ax=ax)
                    st.pyplot(fig)
                else:
                    st.write("No common words to display.")
                
                st.header("Word Cloud")
                # Note: WordCloud generates an image, so we still use matplotlib to display it.
                df_wc = helper.create_wordcloud(selected_user, df)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                plt.axis("off")
                st.pyplot(fig)

                st.header("Emoji Analysis")
                emoji_df = helper.emoji_helper(selected_user, df)
                if not emoji_df.empty:
                    st.dataframe(emoji_df)
                else:
                    st.write("No emojis found.")

            # --- Tab 4: Advanced Analysis ---
            with tab4:
                if selected_user == 'Overall':
                    st.title("Advanced Group Analysis")

                    st.header("Monthly Sentiment Timeline")
                    sentiment_timeline_df = helper.sentiment_timeline(selected_user, df)
                    if not sentiment_timeline_df.empty:
                        # --- NEW Plotly Code ---
                        fig = px.line(sentiment_timeline_df, x='time', y='sentiment_score', markers=True,
                                      title="Average Monthly Sentiment", labels={'sentiment_score': 'Sentiment Score'})
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)

                    st.header("Sentiment by User")
                    user_sentiment_df = helper.user_sentiment_analysis(df)
                    if not user_sentiment_df.empty:
                        # --- NEW Plotly Code ---
                        fig = px.bar(user_sentiment_df, x='sentiment_score', y='user', orientation='h',
                                     color='sentiment_score', color_continuous_scale=px.colors.sequential.Viridis,
                                     title="Average Sentiment Score by User",
                                     labels={'sentiment_score': 'Average Sentiment Score', 'user': 'User'})
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

                    st.header("Interaction Network")
                    # Note: NetworkX graphs are highly specialized and best plotted with matplotlib.
                    interaction_graph, user_counts = helper.create_interaction_network(selected_user, df)
                    if interaction_graph and interaction_graph.number_of_nodes() > 1:
                        fig, ax = plt.subplots(figsize=(15, 15))
                        pos = nx.kamada_kawai_layout(interaction_graph)
                        node_sizes = [user_counts.get(node, 100) * 15 for node in interaction_graph.nodes()]
                        weights = [interaction_graph[u][v]['weight'] for u, v in interaction_graph.edges()]
                        nx.draw(interaction_graph, pos, with_labels=True, node_size=node_sizes,
                                node_color='skyblue', font_size=10, font_weight='bold',
                                width=[w * 0.3 for w in weights], edge_color='grey', alpha=0.7,
                                arrowsize=20, ax=ax)
                        st.pyplot(fig)
                    
                    st.header("Average Response Times (minutes)")
                    response_df = helper.calculate_response_times(selected_user, df)
                    if not response_df.empty:
                        # --- NEW Plotly Code ---
                        fig = px.bar(response_df, x='avg_response_time_minutes', y='user', orientation='h',
                                     color='avg_response_time_minutes', color_continuous_scale=px.colors.sequential.Magma,
                                     title="Average Response Time by User",
                                     labels={'avg_response_time_minutes': 'Response Time (minutes)', 'user': 'User'})
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    #st.header("Discovered Conversation Topics (BERTopic)")
                    #bertopic_info_df, topic_model, topics = helper.find_topics_bertopic(df)
                    #if not bertopic_info_df.empty:
                    #    st.dataframe(bertopic_info_df[bertopic_info_df['Topic'] != -1], hide_index=True)

                    #    st.header("Topic Popularity Over Time")
                    #    timeline_fig = helper.visualize_topics_over_time(df, topic_model, topics)
                    #    if timeline_fig:
                    #        st.plotly_chart(timeline_fig)
                    #else:
                    #    st.info("Could not find distinct topics...")
                else:
                    st.info("Advanced analysis is only available for the 'Overall' group view.")
            # --- Tab 5: Conversation Summarizer ---
            with tab5:
                st.header("Summarize Conversations")
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=df['date'].min().date(), key='summary_start_date')
                with col2:
                    end_date = st.date_input("End Date", value=df['date'].max().date(), key='summary_end_date')

                if st.button("Generate Summary"):
                    progress_text = "Analyzing messages and crafting your summary..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    with st.spinner("Model is processing... Please wait."):
                        summary = helper.generate_summary(df, start_date, end_date, my_bar)
                        my_bar.progress(1.0, text="Summary Generated!")
                        st.success("Summary Generated!")
                        st.text_area("Conversation Summary:", summary, height=250)
                else:
                    st.info("Select your desired date range and click 'Generate Summary' to get an overview of the conversations.")
            # --- Tab 6: Chatbot Q&A ---
            with tab6:
                st.header("Ask Questions About Your Chat")
                
                # Create the chain only if it hasn't been created yet
                if 'qa_chain' not in st.session_state:
                    st.session_state.qa_chain = None
                    st.session_state.vector_store = None

                # Button to initialize the chatbot (this is a one-time process per chat)
                if st.button("Initialize Chatbot"):
                    with st.spinner("Analyzing chat and preparing the chatbot..."):
                        st.session_state.vector_store, st.session_state.qa_chain = helper.create_conversational_chain(df)
                        st.success("Chatbot is ready!")

                if st.session_state.qa_chain:
                    user_question = st.text_input("Your Question:")
                    if user_question:
                        with st.spinner("Finding the answer..."):
                            docs = st.session_state.vector_store.similarity_search(user_question)
                            response = st.session_state.qa_chain(
                                {"input_documents": docs, "question": user_question},
                                return_only_outputs=True
                            )
                            st.markdown("### Answer:")
                            st.write(response["output_text"])
                else:
                    st.info("Click 'Initialize Chatbot' to start asking questions about your chat data.")


else:
    st.info("Awaiting for a WhatsApp chat file to be uploaded.")