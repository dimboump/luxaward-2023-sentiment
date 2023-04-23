# LUX Audience Award Sentiment Analysis Demo App

This is a demo Streamlit app showcasing sentiment analysis capabilities on short movie reviews for the 2023 LUX Audience Award.

It utilizes [HuggingFace's transformers](https://github.com/huggingface/transformers) library for sentiment analysis, and displays the results through a word cloud and a bar chart. The app also stores the reviews on an Amazon S3 bucket for persistence.

> Please note that this app is not affiliated with the LUX Audience Award event in any way.
> Go to [https://luxaward.eu](https://luxaward.eu) to rate the movies!

## Run the app

### Online

The app is hosted on Streamlit and can be accessed at [https://luxaward-2023-sentiment.streamlit.app/](https://luxaward-2023-sentiment.streamlit.app/).

### Locally

1. Clone the repository:
   1. SSH (recommended): `git clone git@github.com:dimboump/luxaward-2023-sentiment/`.
   2. HTTPS: `git clone https://dimboump.github.com/luxaward-2023-sentiment/`.
2. Install the required dependencies with the command `pip install -r requirements.txt`.
3. Run `streamlit run app.py` to start the app in `localhost`.
