import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# --- Streamlit UI Setup ---
st.sidebar.title("WhatsApp Chat Analyzer")
st.sidebar.markdown("Analyze your WhatsApp chats to gain insights into your conversations.")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat export file (.txt)")

if uploaded_file is not None:
    # Read the file
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    
    # Preprocess the data
    df = preprocessor.preprocess(data)

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

            # Display a snippet of the processed data
            st.markdown("### Processed Data Snippet")
            st.dataframe(df.head())

            # --- Top Statistics ---
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

            # --- Timelines ---
            st.header("Monthly Activity")
            timeline = helper.monthly_timeline(selected_user, df)
            if not timeline.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(x='time', y='message', data=timeline, marker='o', color='green', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            # --- Activity Maps ---
            col1, col2 = st.columns(2)
            with col1:
                st.header("Busiest Day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                sns.barplot(x=busy_day.index, y=busy_day.values, palette="rocket", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            with col2:
                st.header("Busiest Month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                sns.barplot(x=busy_month.index, y=busy_month.values, palette="cubehelix", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
            # --- Activity Heatmap ---
            st.header("Weekly Activity Heatmap")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            if not user_heatmap.empty:
                fig, ax = plt.subplots(figsize=(15, 8))
                sns.heatmap(user_heatmap, cmap="YlGnBu", ax=ax)
                st.pyplot(fig)

            # --- Word Analysis ---
            st.header("Most Common Words")
            most_common_df = helper.most_common_words(selected_user, df)
            if not most_common_df.empty:
                fig, ax = plt.subplots()
                sns.barplot(x=most_common_df[1], y=most_common_df[0], palette="plasma", ax=ax)
                st.pyplot(fig)
            else:
                st.write("No common words to display.")
                
            # --- Emoji Analysis ---
            st.header("Emoji Analysis")
            emoji_df = helper.emoji_helper(selected_user, df)
            if not emoji_df.empty:
                st.dataframe(emoji_df)
            else:
                st.write("No emojis found.")

            # --- Advanced Analysis (Only for 'Overall' view) ---
            if selected_user == 'Overall':
                st.title("Advanced Group Analysis")

                # Sentiment Timeline
                st.header("Monthly Sentiment Timeline")
                sentiment_timeline_df = helper.sentiment_timeline(selected_user, df)
                if not sentiment_timeline_df.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(x='time', y='sentiment_score', data=sentiment_timeline_df, marker='o', color='purple', ax=ax)
                    plt.axhline(y=0, color='red', linestyle='--')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                # Interaction Network
                st.header("Interaction Network")
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
                
                # Response Times
                st.header("Average Response Times (minutes)")
                response_df = helper.calculate_response_times(selected_user, df)
                if not response_df.empty:
                    fig, ax = plt.subplots()
                    sns.barplot(x='avg_response_time_minutes', y='user', data=response_df, palette='magma', ax=ax)
                    st.pyplot(fig)
            # --- Agentic AI Chatbot Section ---
            st.title("Chat with Your Data 🤖")
            st.markdown("Ask a question in plain English and let the AI agent analyze the chat for you.")

            # Get user's question
            user_question = st.text_input("e.g., Who sent the most emojis? or What was the overall sentiment in June 2025?")

            if st.button("Ask AI Agent"):
                if user_question:
                    with st.spinner("The AI agent is analyzing..."):
                        # Pass the entire DataFrame to the helper function
                        response = helper.get_gemini_response(df, user_question)
                        st.markdown("### Agent's Answer:")
                        st.write(response)
                else:
                    st.warning("Please ask a question.")

else:
    st.info("Awaiting for a WhatsApp chat file to be uploaded.")

