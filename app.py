import sys
import os
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def run_analysis():
    if len(sys.argv) < 2:
        print("Error: Please provide the path to your WhatsApp chat file.")
        print("Usage: python app.py \"path/to/chat.txt\" [\"Optional User Name\"]")
        return

    filepath = sys.argv[1]
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: The file was not found at the path: {filepath}")
        return

    df = preprocessor.preprocess(data)
    if df.empty:
        print("Could not parse any messages from the chat file. Exiting.")
        return

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    
    print("\nAvailable users for analysis:")
    for user in user_list:
        print(f"- {user}")
    
    if len(sys.argv) > 2:
        selected_user = sys.argv[2]
        if selected_user not in user_list:
            print(f"\nError: User '{selected_user}' not found in the chat.")
            return
    else:
        selected_user = 'Overall'

    output_folder = "analysis_results"
    os.makedirs(output_folder, exist_ok=True)
    sns.set_theme(style="darkgrid")

    print(f"\n📈 WHATSAPP CHAT ANALYSIS FOR: {selected_user} 📈")
    print("==================================================")

    # 1. Top Statistics
    num_messages, words, num_media, num_links = helper.fetch_stats(selected_user, df)
    print("\n## Top Statistics ##")
    print(f"Total Messages: {num_messages}\nTotal Words: {words}\nMedia Shared: {num_media}\nLinks Shared: {num_links}")

    # 2. Most Busy Users (Plot only for 'Overall')
    if selected_user == 'Overall':
        print("\n## Most Busy Users ##")
        x, user_percent_df = helper.most_busy_users(df)
        x = x[x.index != 'group_notification']
        print(user_percent_df)
        if not x.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=x.index, y=x.values, palette="viridis")
            plt.title('Most Busy Users'); plt.xlabel('User'); plt.ylabel('Number of Messages')
            plt.xticks(rotation=45); plt.tight_layout()
            save_path = os.path.join(output_folder, 'most_busy_users.png')
            plt.savefig(save_path); print(f"--> Plot saved to {save_path}"); plt.close()

    # 3. Most Common Words
    print("\n## Most Common Words ##")
    most_common_df = helper.most_common_words(selected_user, df)
    print(most_common_df)
    if not most_common_df.empty:
        plt.figure(figsize=(10, 8))
        sns.barplot(x=most_common_df[1], y=most_common_df[0], palette="plasma")
        plt.title('Top 20 Most Common Words'); plt.xlabel('Count'); plt.ylabel('Words')
        plt.tight_layout(); save_path = os.path.join(output_folder, 'most_common_words.png')
        plt.savefig(save_path); print(f"--> Plot saved to {save_path}"); plt.close()

    # 4. Emoji Analysis
    print("\n## Emoji Analysis (Top 10) ##")
    emoji_df = helper.emoji_helper(selected_user, df)
    print(emoji_df.head(10))
    if not emoji_df.empty:
        plt.figure(figsize=(10, 6)); sns.barplot(x=emoji_df[1].head(10), y=emoji_df[0].head(10), palette="coolwarm")
        plt.title('Top 10 Most Used Emojis'); plt.xlabel('Count'); plt.ylabel('Emoji')
        plt.tight_layout(); save_path = os.path.join(output_folder, 'emoji_analysis.png')
        plt.savefig(save_path); print(f"--> Plot saved to {save_path}"); plt.close()

    # 5. Monthly Activity Timeline
    print("\n## Monthly Activity ##")
    monthly_timeline = helper.monthly_timeline(selected_user, df)
    print(monthly_timeline)
    if not monthly_timeline.empty:
        plt.figure(figsize=(12, 6)); sns.lineplot(x='time', y='message', data=monthly_timeline, marker='o', color='green')
        plt.title('Monthly Chat Activity'); plt.xlabel('Month-Year'); plt.ylabel('Number of Messages')
        plt.xticks(rotation=45); plt.tight_layout(); save_path = os.path.join(output_folder, 'monthly_activity.png')
        plt.savefig(save_path); print(f"--> Plot saved to {save_path}"); plt.close()
    
    # 6. Busiest Day and Month
    print("\n## Busiest Day of the Week ##")
    busy_day = helper.week_activity_map(selected_user, df)
    print(busy_day)
    if not busy_day.empty:
        plt.figure(figsize=(10, 6)); sns.barplot(x=busy_day.index, y=busy_day.values, palette="rocket")
        plt.title('Busiest Day of the Week'); plt.xlabel('Day'); plt.ylabel('Number of Messages')
        plt.xticks(rotation=45); plt.tight_layout(); save_path = os.path.join(output_folder, 'busiest_day.png')
        plt.savefig(save_path); print(f"--> Plot saved to {save_path}"); plt.close()

    print("\n## Busiest Month ##")
    busy_month = helper.month_activity_map(selected_user, df)
    print(busy_month)
    if not busy_month.empty:
        plt.figure(figsize=(10, 6)); sns.barplot(x=busy_month.index, y=busy_month.values, palette="cubehelix")
        plt.title('Busiest Month'); plt.xlabel('Month'); plt.ylabel('Number of Messages')
        plt.xticks(rotation=45); plt.tight_layout(); save_path = os.path.join(output_folder, 'busiest_month.png')
        plt.savefig(save_path); print(f"--> Plot saved to {save_path}"); plt.close()

    # 7. Sentiment Timeline
    print("\n## Sentiment Timeline ##")
    sentiment_timeline_df = helper.sentiment_timeline(selected_user, df)
    if not sentiment_timeline_df.empty:
        plt.figure(figsize=(12, 6)); sns.lineplot(x='time', y='sentiment_score', data=sentiment_timeline_df, marker='o', color='purple')
        plt.title('Monthly Sentiment Timeline'); plt.xlabel('Month-Year'); plt.ylabel('Average Sentiment Score')
        plt.axhline(y=0, color='red', linestyle='--'); plt.xticks(rotation=45); plt.tight_layout()
        save_path = os.path.join(output_folder, 'sentiment_timeline.png')
        plt.savefig(save_path); print(f"--> Plot saved to {save_path}"); plt.close()

    # 8. Activity Heatmap
    print("\n## Weekly Activity Heatmap ##")
    user_heatmap = helper.activity_heatmap(selected_user, df)
    if not user_heatmap.empty:
        plt.figure(figsize=(20, 10)); sns.heatmap(user_heatmap, cmap="YlGnBu")
        plt.title('Weekly Activity Heatmap (Day vs. Hour)'); plt.xlabel('Time Period (24-Hour Format)'); plt.ylabel('Day of the Week')
        plt.tight_layout(); save_path = os.path.join(output_folder, 'activity_heatmap.png')
        plt.savefig(save_path); print(f"--> Plot saved to {save_path}"); plt.close()

    # 9. Topic Modeling
    print("\n## Top 5 Conversation Topics ##")
    try:
        topics = helper.topic_modeling(selected_user, df)
        for topic_id, topic_words in topics:
            print(f"Topic #{topic_id + 1}: {topic_words}")
    except Exception as e:
        print(f"Could not perform topic modeling. Reason: {e}")
        
    # 10. Named Entity Recognition
    print("\n## Most Mentioned Entities ##")
    persons_df, places_df, orgs_df = helper.named_entity_recognition(selected_user, df)
    print("\n--- Top 10 People Mentioned ---\n", persons_df)
    print("\n--- Top 10 Places Mentioned ---\n", places_df)
    print("\n--- Top 10 Organizations Mentioned ---\n", orgs_df)
    
    # 11. Interaction Network Graph
    print("\n## Interaction Network ##")
    interaction_graph, user_counts = helper.create_interaction_network(selected_user, df)

    if interaction_graph and interaction_graph.number_of_nodes() > 1:
        plt.figure(figsize=(18, 18)) # Make the figure larger
        
        # Use Kamada-Kawai layout for a more aesthetically pleasing arrangement
        pos = nx.kamada_kawai_layout(interaction_graph)
        
        # --- Node size based on message count ---
        node_sizes = [user_counts.get(node, 100) * 15 for node in interaction_graph.nodes()]

        # --- Edge width based on interaction weight ---
        weights = [interaction_graph[u][v]['weight'] for u, v in interaction_graph.edges()]
        
        # Draw the graph with improved aesthetics
        nx.draw(interaction_graph, pos, with_labels=True, 
                node_size=node_sizes,
                node_color='skyblue', 
                font_size=12, 
                font_weight='bold',
                width=[w * 0.3 for w in weights], # Slightly thicker edges
                edge_color='grey', 
                alpha=0.7, # Use transparency for edges
                arrowsize=25)
                
        plt.title('Interaction Network', size=20)
        save_path = os.path.join(output_folder, 'interaction_network.png')
        plt.savefig(save_path, bbox_inches='tight') # Use bbox_inches to prevent labels from being cut off
        print(f"--> Plot saved to {save_path}")
        plt.close()
            
        # 12. Average Response Times
        print("\n## Average Response Time (in minutes) ##")
        response_df = helper.calculate_response_times(selected_user, df)
        print(response_df)
        if not response_df.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='avg_response_time_minutes', y='user', data=response_df, palette='magma')
            plt.title('Average User Response Time')
            plt.xlabel('Average Time (minutes)'); plt.ylabel('User')
            plt.tight_layout()
            save_path = os.path.join(output_folder, 'response_times.png')
            plt.savefig(save_path); print(f"--> Plot saved to {save_path}"); plt.close()

    print("\n==================================================")
    print(f"Analysis Complete. All plots have been saved to the '{output_folder}' folder.")

if __name__ == "__main__":
    run_analysis()