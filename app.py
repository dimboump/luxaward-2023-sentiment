import json
from typing import Any, Dict, List, cast

import pandas as pd
import s3fs
import streamlit as st
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, pipeline
from wordcloud import WordCloud

MOVIES = ['Alcarràs', 'Burning Days', 'Close',
          'Triangle of Sadness', 'Will-o-the-Wisp']
MODEL = 'nlptown/bert-base-multilingual-uncased-sentiment'
BUCKET_NAME = "luxaward"
REVIEWS_FILE = "movie_reviews.json"


fs = s3fs.S3FileSystem(anon=False)


def get_sentiment(review: str) -> int:
    classifier = pipeline('sentiment-analysis',
                          model=MODEL,
                          tokenizer=AutoTokenizer.from_pretrained(MODEL))
    sentiment_score = classifier(review)[0]['label']
    return int(sentiment_score[0])


def generate_wordcloud(movie_reviews: str) -> None:
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10).generate(movie_reviews)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=None)
    ax.imshow(wordcloud)
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig)


@st.cache_data(ttl=600)
def load_reviews() -> Dict[str, List[Any]]:
    with fs.open(f'{BUCKET_NAME}/{REVIEWS_FILE}', 'r') as f:
        reviews = json.load(f)
    return reviews


def save_reviews(reviews: Dict[str, list]) -> None:
    with fs.open(f'{BUCKET_NAME}/{REVIEWS_FILE}', 'w') as f:
        json.dump(reviews, f)


def app() -> None:
    st.set_page_config(
        page_title='Sentiment Analysis on LUX Audience Award Movie Reviews',
        page_icon='🎬',
        layout='wide',
    )
    st.title('LUX Audience Award Movie Reviews')
    st.subheader('Sentiment analysis of movie reviews')
    st.error("""This is a demo app. 
             It is by no means affiliated with the LUX Audience Award. 
             Rate the movies at https://luxaward.eu/.""")

    col1, col2 = st.columns(2)

    with col1:
        movie_choice = st.selectbox('Select movie', MOVIES)
        movie_choice = cast(str, movie_choice)

        review = st.text_area(
            'Write your review (max 1000 characters)', max_chars=1000)

        while len(review) > 1000:
            st.warning('Review too long. Please limit to 1000 characters.')

        while len(review) > 0:
            sentiment_score = get_sentiment(review)
            st.write("Your review's sentiment:")
            emoji = '😄' if sentiment_score > 3 else '😞' \
                    if sentiment_score < 3 else '😐'
            st.markdown(f'<p style="font-size: 3.5rem;">{emoji}</p>',
                        unsafe_allow_html=True)
            break

        if st.button('Submit review'):
            if len(review) > 0:
                reviews = load_reviews()
                reviews.setdefault(movie_choice, [])
                reviews[movie_choice] = cast(List[Any], reviews[movie_choice])
                reviews[movie_choice].append({"review": review, "sentiment_score": sentiment_score})
                save_reviews(reviews)
                st.success('Review submitted successfully!')
            else:
                st.warning('Please enter a review.')

    with col2:
        st.subheader('Wordcloud for ' + movie_choice)
        reviews = load_reviews()
        movie_reviews = reviews.get(movie_choice, [])
        movie_reviews_text = ' '.join([r["review"] for r in movie_reviews])

        if len(movie_reviews_text) > 0:
            generate_wordcloud(movie_reviews_text)
        else:
            st.warning('No reviews found for this movie. Write the first one!')

        # Generate plot of sentiment analysis for all movies
        st.subheader('Sentiment for all movies')
        reviews = load_reviews()
        df = pd.DataFrame(columns=['movie', 'sentiment'])
        for movie, movie_reviews in reviews.items():
            for review in movie_reviews:
                df = df.append({
                    'movie': movie,
                    'sentiment': review.get('sentiment_score', None)
                }, ignore_index=True)
        if not df.empty:
            try:
                st.bar_chart(df.groupby(['movie', 'sentiment']).size())
            except KeyError:
                st.warning('No reviews found.')
        else:
            st.warning('No reviews found.')


if __name__ == '__main__':
    app()
