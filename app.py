import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd
import base64
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
from helper import perform_sentiment_analysis, plot_sentiment_scores
from tempfile import NamedTemporaryFile
from fpdf import FPDF
# Suppressing the Streamlit warning
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_css(css_path):
    with open(css_path, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css("style.css")

# Main header and logo
st.image("images/logoig.jpeg", use_column_width=True)
st.title('WhatsApp Chat Analyzer')

st.sidebar.image("images/newlogo.jpeg", use_column_width=True)
uploaded_file = st.sidebar.file_uploader("Upload Exported Chat", type=["txt", "csv"])

figs = []


def generate_blank_pdf(filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="This is a blank PDF", ln=True, align='C')
    pdf_output_path = f"{filename}.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path


# Function to generate PDF report
def generate_pdf_report(figs, titles, filename, selected_user):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf_output_path = tmpfile.name

        stats = {
            'Total Messages': helper.fetch_stats(selected_user, df)[0],
            'Total words': helper.fetch_stats(selected_user, df)[1],
            'Media shared': helper.fetch_stats(selected_user, df)[2],
            'Links shared': helper.fetch_stats(selected_user, df)[3],
            'Emoji shared': helper.fetch_stats(selected_user, df)[4],
            'Deleted Messages': helper.fetch_stats(selected_user, df)[5],
            'Edited Messages': helper.fetch_stats(selected_user, df)[6],
            'Contact shared': helper.fetch_stats(selected_user, df)[7]
        }

        # Add title to the PDF
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Analysis Report", ln=True, align='C')
        pdf.cell(100, 10, txt="User Statistics", ln=True, align='L')
        pdf.ln(10)

        # Add statistics to the PDF
        pdf.set_font("Arial", size=12)
        for key, value in stats.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align='L')
        pdf.ln(10)

        # Add figures to the PDF
        for i in range(0, len(figs), 2):
            pdf.add_page()  # Add a new page for each pair of figures

            # Add the first pair of figures
            fig1 = figs[i]
            title1 = titles[i]
            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile1:
                fig1.savefig(tmpfile1.name)
                pdf.image(tmpfile1.name, x=pdf.get_x(), y=pdf.get_y() + 10, w=180, h=100)
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt=title1, ln=True, align='C')
                pdf.ln(110)  # Move to the next line after each plot

            # Calculate the horizontal position for the second figure
            second_figure_x = pdf.get_x() + 100 if pdf.get_x() + 100 + 180 <= pdf.w else pdf.l_margin

            # Check if there's a second figure in the pair
            if i + 1 < len(figs):
                fig2 = figs[i + 1]
                title2 = titles[i + 1]
                with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile2:
                    fig2.savefig(tmpfile2.name)
                    pdf.image(tmpfile2.name, x=second_figure_x, y=pdf.get_y() + 10, w=180, h=100)
                    pdf.set_font("Arial", size=14)
                    pdf.cell(200, 10, txt=title2, ln=True, align='C')
                    pdf.ln(110)  # Move to the next line after each plot

        pdf.output(pdf_output_path)

    return pdf_output_path


# Function to create download link
def create_download_link_sidebar(val, filename):
    b64 = base64.b64encode(val).decode()  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.pdf">Download PDF file</a>'

if uploaded_file:
    # To read file as bytes
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique Users
    user_list = df['username'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall Users")

    # Adding a search box for selecting the user
    search_user = st.sidebar.text_input("Search for a User", "")

    # Filter the user list based on the search input
    filtered_users = [user for user in user_list if search_user.lower() in user.lower()]

    # Creating a select box with the filtered user list
    selected_user = st.sidebar.selectbox("Select The User", filtered_users)
    if selected_user == "Overall Users":
        analysis_menu = ["User Statistics","Sentiment Analysis","Comparative Analysis","User Activity","Overall User Activity",
                        "Word and Emoji Analysis","Timeline Analysis"]
    else:
        analysis_menu = ["User Statistics","Sentiment Analysis","Comparative Analysis","User Activity","Overall User Activity",
                        "Word and Emoji Analysis","Timeline Analysis"]
    st.sidebar.header("Analysis Options")
    choice = st.sidebar.selectbox("Select Analysis Type", analysis_menu, index=0)

    # Sentiment Analysis
    if choice == "Sentiment Analysis":

        # Choose sentiment analysis method
        #method = st.sidebar.selectbox("Choose sentiment analysis method", ["textblob", "vader"])

        #method = st.sidebar.selectbox("Choose sentiment analysis method", ["vader"], index=0, key="method_select")
        #st.markdown("<style>div.row-widget.stRadio > div{visibility: hidden;}</style>", unsafe_allow_html=True)

        # Update sentiment analysis button handler
        if st.sidebar.button("Show Sentiment Analysis", key="sentiment_analysis_button"):
            if selected_user != 'Overall Users':
                df = df[df['username'] == selected_user]
            user_sentiment_scores = perform_sentiment_analysis(df)
            st.title('Sentimental Analysis')

            # Plot overall sentiment scores
            plot = plot_sentiment_scores(user_sentiment_scores)
            # Display overall sentiment plot
            st.pyplot(plot)
            figs.append(plot)  # Append the plot to figs list

            # Display overall sentiment scores DataFrame
            st.write(user_sentiment_scores)
        else:
            # Check if user_sentiment_scores is defined
            if 'user_sentiment_scores' in locals():
                # Check if there are rows matching the filter condition for the selected user
                if not user_sentiment_scores[user_sentiment_scores['username'] == selected_user].empty:
                    # Display sentiment score for the selected user
                    selected_user_sentiment_score = \
                        user_sentiment_scores[user_sentiment_scores['username'] == selected_user][
                            'average_sentiment_score'].iloc[0]
                    st.title(f"Sentiment Score for {selected_user}")
                    st.write(selected_user_sentiment_score)
                else:
                    st.write(f"No sentiment score available for {selected_user}.")
            else:
                st.write("Click the 'Show Sentiment Analysis' button to calculate sentiment scores.")

    elif choice == "Comparative Analysis":

        st.subheader("Comparative Analysis between Users")

        users_to_compare = st.multiselect("Select users for comparison", user_list)

        st.write("---")

        if users_to_compare:

            min_date = df["date"].min().date()

            max_date = df["date"].max().date()

            selected_range = st.slider("Select Time Range", min_date, max_date, (min_date, max_date))

            st.write("---")

            # Convert the selected_dates to numpy datetime64

            start_date = np.datetime64(selected_range[0])

            end_date = np.datetime64(selected_range[1] + pd.Timedelta(days=1))  # To include the end date in the range

            date_filtered_df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date < end_date)]

            user_filtered_df = date_filtered_df[date_filtered_df["username"].isin(users_to_compare)]

            # Debug statements

            st.write("Filtered Data within Selected Time Range:")

            st.write(date_filtered_df.head())  # Print first few rows of filtered data

            st.write("Filtered Data for Selected Users:")

            st.write(user_filtered_df.head())  # Print first few rows of user filtered data

            users_activity = user_filtered_df["username"].value_counts()

            # Debug statement

            st.write("Users Activity:")

            st.write(users_activity)  # Print users_activity to check its contents


            plt.bar(users_activity.index, users_activity.values)
            # Create a new figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot the bar chart on the created figure
            plt.bar(users_activity.index, users_activity.values)
            plt.xlabel('Users')
            plt.ylabel('Message Count')
            plt.title('Comparative Analysis: Message Count by Users')
            plt.xticks(rotation=45)

            # Display the bar chart in Streamlit
            st.pyplot(fig)

            # Append the created figure to figs
            figs.append(fig)


    elif st.sidebar.button("Start Analysis"):
        # User Statistics
        if choice == "User Statistics":
            # Fetching stats
            total_messages, total_words, total_media_messages, total_url, total_emoji, deleted_message, edited_messages, shared_contact = helper.fetch_stats(
                selected_user, df)
            st.markdown("### Total Messages Shared: ")
            st.write(f"<div class='big-font'>{total_messages}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Words Shared: ")
            st.write(f"<div class='big-font'>{total_words}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Media Shared: ")
            st.write(f"<div class='big-font'>{total_media_messages}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Link Shared: ")
            st.write(f"<div class='big-font'>{total_url}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Emoji Shared: ")
            st.write(f"<div class='big-font'>{total_emoji}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Deleted Message: ")
            st.write(f"<div class='big-font'>{deleted_message}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Edited Message: ")
            st.write(f"<div class='big-font'>{edited_messages}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Contact Shared: ")
            st.write(f"<div class='big-font'>{shared_contact}</div>", unsafe_allow_html=True)

            #st.write("---")

            #st.markdown("### Total Location Shared: ")
            #st.write(f"<div class='big-font'>{shared_location}</div>", unsafe_allow_html=True)

        # User Activity
        elif choice == "User Activity":
            if selected_user == 'Overall Users':
                top, bottom = helper.most_least_busy_users(df)

                st.subheader('Most Active Users')
                st.bar_chart(top)

                st.write("---")

                st.subheader('Least Active Users')
                st.bar_chart(bottom)

                st.write("---")

            # Week Activity Map
            st.subheader("Week Activity Map")
            week_activity_data = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 6))
            week_activity_data.sort_index().plot(kind='bar', ax=ax)
            ax.set_title("Activity Throughout the Week")
            ax.set_ylabel("Number of Messages")
            ax.set_xlabel("Day of the Week")
            st.pyplot(fig)
            figs.append(fig)

            st.write("---")

            # Month Activity Map
            st.subheader("Month Activity Map")
            month_activity_data = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 6))
            month_activity_data.sort_index().plot(kind='bar', ax=ax)
            ax.set_title("Activity Throughout the Month")
            ax.set_ylabel("Number of Messages")
            ax.set_xlabel("Month")
            st.pyplot(fig)
            figs.append(fig)

            st.write("---")

            # Activity Heatmap
            st.subheader("Activity Heatmap")
            heatmap_data = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".0f", ax=ax)
            ax.set_title("Activity Heatmap: Day vs. Period")
            st.pyplot(fig)
            figs.append(fig)

            # Adding another column for message length; using the apply method
            df['message_length'] = df['message'].apply(lambda x: len(x))

            # Grouping by 'username' and calculating the average message length for each user
            avg_msg_lengths = df.groupby('username')['message_length'].mean().reset_index()

            # Selecting the top 5 users based on average message length
            top5_users = avg_msg_lengths.sort_values(by='message_length', ascending=False).head(10)

            # Plotting the bar graph
            plt.figure(figsize=(10, 6))
            st.subheader("Top 10 Users by Average Message Length")

            # Creating the plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=top5_users, x='username', y='message_length', palette='viridis')
            plt.title('Top 5 Users by Average Message Length')
            plt.xlabel('User')
            plt.ylabel('Average Message Length')
            plt.xticks(rotation=45)

            # Displaying the plot using Streamlit
            st.pyplot()


            st.write("Average Message Lengths:")
            for index, row in top5_users.iterrows():
                st.write(f"{row['username']}: {row['message_length']:.2f} characters")

        elif choice == "Overall User Activity":
            user_activity_df = helper.user_activity_in_chat(df)
            st.header("Activity Analysis of Each User in a Group Chat:")

            st.subheader("Total Messages Sent By The User:")
            st.bar_chart(user_activity_df, x='username', y='Total_Messages')

            st.write("---")

            st.subheader("Total Words Sent By The User:")
            st.bar_chart(user_activity_df, x='username', y='Total_Words')

            st.write("---")

            st.subheader("Percentage of Messages Sent By The User:")
            st.bar_chart(user_activity_df, x='username', y='Percentage')

            st.write("---")

            st.subheader("Total Media Shared By The User:")
            st.bar_chart(user_activity_df, x='username', y='Media_Shared')

            st.write("---")

            st.subheader("Total Links Shared By The User:")
            st.bar_chart(user_activity_df, x='username', y='Links_Shared')

            st.write("---")

            st.subheader("Total Emojis Shared By The User:")
            st.bar_chart(user_activity_df, x='username', y='Emojis_Shared')

            st.write("---")

            st.subheader("Total Deleted Messages By Each User:")
            st.bar_chart(user_activity_df, x='username', y='Deleted_Messages')

            st.write("---")

            st.subheader("Total Edited Messages By Each User:")
            st.bar_chart(user_activity_df, x='username', y='Edited_Messages')

            st.write("---")

            st.subheader("Total Contacts Shared By The User:")
            st.bar_chart(user_activity_df, x='username', y='Shared_Contacts')

            st.write("---")

        # Word and Emoji Analysis
        elif choice == "Word and Emoji Analysis":
            wordcloud_image = helper.create_wordcloud(selected_user, df)
            # Convert WordCloud to Image
            wc_img = Image.new("RGB", (wordcloud_image.width, wordcloud_image.height))
            wc_array = np.array(wordcloud_image)
            wc_img.paste(Image.fromarray(wc_array), (0, 0))
            st.subheader("Word Cloud:")
            st.image(wc_img, use_column_width=True, caption="Word Cloud of Chat")
            figs.append(wc_img)
            st.write("---")

            # Emoji Analysis
            emoji_df = helper.emoji_helper(selected_user, df)
            st.subheader("Emoji Analysis:")
            st.dataframe(emoji_df)
            st.write("---")

            # Plot top 5 emojis
            plt.rcParams['font.sans-serif'] = ['Segoe UI Emoji']
            fig, ax = plt.subplots()
            ax.pie(emoji_df['Frequency'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f")
            st.pyplot(fig)
            figs.append(fig)

        # Timeline Analysis
        elif choice == "Timeline Analysis":
            monthly_timeline = helper.monthly_timeline(selected_user, df)
            daily_timeline = helper.daily_timeline(selected_user, df)

            st.subheader("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation=60)  # Rotate the x-axis tick labels
            st.pyplot(fig)
            figs.append(fig)
            st.write("---")

            st.subheader("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['date'], daily_timeline['message'], color='skyblue')
            plt.xticks(rotation=60)
            st.pyplot(fig)
            figs.append(fig)

#st.sidebar.markdown("---")
#st.sidebar.markdown("© 2024 ChatVerse. All rights reserved.")

# Define the titles for overall analysis
overall_titles = ["User Statistics","Sentiment Analysis","Comparative Analysis","User Activity","Overall User Activity",
                        "Word and Emoji Analysis","Timeline Analysis"]

# Define the titles for selected user analysis
selected_user_titles = ["User Statistics","Sentiment Analysis","Comparative Analysis","User Activity","Overall User Activity",
                        "Word and Emoji Analysis","Timeline Analysis"]

# Determine which titles to use based on the selected user
if selected_user == 'Overall':
    titles = overall_titles
else:
    titles = selected_user_titles

# Filter out any titles that exceed the number of available figures
titles = titles[:len(figs)]

# Remove the sentiment score title if it's available and the user is not "Overall"
if selected_user != 'Overall' and "Sentiment Scores" in titles:
    titles.remove("Sentiment Scores")

# Generate PDF report
pdf_output_path = generate_pdf_report(figs, titles, "testfile", selected_user)

# Read the generated PDF
with open(pdf_output_path, "rb") as f:
    pdf_bytes = f.read()

# Create download link with the filename including titles
download_link_sidebar = create_download_link_sidebar(pdf_bytes, "Analysis_Report")

# Display the download link in the sidebar
st.sidebar.markdown(download_link_sidebar, unsafe_allow_html=True)