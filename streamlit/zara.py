import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging
import os
import boto3
from botocore.exceptions import ClientError
import seaborn as sns

from collections import defaultdict
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

plt.tight_layout()
plt.style.use('dark_background')

s3 = boto3.client('s3', aws_access_key_id='AKIA25ZCC7LD4EQX2M4F',
                  aws_secret_access_key='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')


def load_file(file: str, type='json'):
    obj = s3.get_object(Bucket='tweetstfm', Key=file)
    if type == 'json':
        tweets = pd.read_json(obj.get("Body"))
        return tweets
    if type == 'csv':
        tweets = pd.read_csv(obj.get("Body"))
        return tweets


def upload_file(file_name, bucket='tweetstfm', object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)
    try:
        response = s3.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


stop_words = list(set(stopwords.words("english")))


def generate_ngrams(serie: pd.Series, n_gram=1):
    def tokens(text, n=1):
        token = [token for token in text.lower().split(' ') if token != '' if token not in stop_words]
        ngrams = zip(*[token[i:] for i in range(n)])
        return [' '.join(ngram) for ngram in ngrams]

    ngram_words = defaultdict(int)
    for tweet in serie.values:
        for word in tokens(tweet, n_gram):
            ngram_words[word] += 1

    df_ngram = pd.DataFrame(sorted(ngram_words.items(), key=lambda x: x[1])[::-1])

    return df_ngram


def plot_ngrams(df_gram1: pd.DataFrame, df_gram2: pd.DataFrame, n: int = 20):
    fig, axes = plt.subplots(ncols=2, figsize=(18, n // 2), dpi=100)
    sns.barplot(y=df_gram1[:n][0], x=df_gram1[:n][1], ax=axes[0], color='green')
    sns.barplot(y=df_gram2[:n][0], x=df_gram2[:n][1], ax=axes[1], color='red')

    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].tick_params(axis='y', labelsize=13)

    return plt


def show_pca(df: pd.DataFrame):
    return px.scatter(x=df['PCA1'], y=df['PCA2'], color=df['topic'])


st.set_page_config(layout="wide")
st.title('Bienvenido a nuestro proyecto ZARA')

st.subheader('Zara Análisis de Sentimientos')
st.write('Como podemos observar como Racismo o Racista que tiene un fuerte impacto en los comentarios')

df_zara_pred_sentimental = load_file('zara_pnl_pred.csv', 'csv')
df_zara_perc_sentimental = load_file('zara_pnl_percent.csv', 'csv')

row1Senti, row2Senti = st.columns(2)

with row1Senti:
    st.dataframe(df_zara_pred_sentimental)

with row2Senti:
    st.dataframe(df_zara_perc_sentimental)

df_zara = load_file('zara_features.csv', 'csv')
df_topic = load_file('topic.csv', 'csv')

st.subheader('Zara tópicos')
st.dataframe(df_zara)
st.write(f'Cantidad de registros obtenidos {df_zara.shape[0]}')

# Start Show Words Gram plots

df_text = df_zara[['comment']].dropna().drop_duplicates()['comment']
df_ngram_1 = generate_ngrams(df_text, 1)
df_ngram_2 = generate_ngrams(df_text, 2)

plot_ngram = plot_ngrams(df_ngram_1, df_ngram_2, 20)

st.subheader('Palabras más concurrentes (UniGram) y (BiGram)')
st.pyplot(plot_ngram)

st.subheader('Topicos')
st.dataframe(df_topic)
st.plotly_chart(show_pca(df_topic), use_container_width=True)
# End

row1, row2, row3 = st.columns(3)

df_comments = df_zara[['comment']].dropna().drop_duplicates()
df_tags = df_zara[['Tags']].dropna().drop_duplicates()
df_emojis = df_zara[['Emojis']].dropna().drop_duplicates()

with row1:
    st.subheader('Comentarios')
    st.dataframe(df_comments)
    st.write(f'Cantidad de comentarios con etiquetas {df_comments.shape[0]}')

with row2:
    st.subheader('Etiquetas')
    st.dataframe(df_tags)
    st.write(f'Cantidad de comentarios con etiquetas {df_tags.shape[0]}')

with row3:
    st.subheader('Emojis')
    st.dataframe(df_emojis)
    st.write(f'Cantidad de comentarios con etiquetas {df_emojis.shape[0]}')

year_input = st.sidebar.selectbox('Año', list(reversed(range(2021, 2023))))


@st.cache
def filter_by_year(year):
    return df_zara[(df_zara['Year'] == year)]


@st.cache
def filter_by_month(year, month):
    return df_zara[(df_zara['Year'] == year) & (df_zara['Month'].isin(month))]


@st.cache
def group_by_date(df: pd.DataFrame):
    return df.groupby(['Date']).agg({
        'comment': 'count',
        'TagsCount': 'sum',
        'EmojiCounts': 'sum'
    }).reset_index()


@st.cache
def filter_by_tags(df: pd.DataFrame, tag: str):
    frame = df.dropna()
    frame = frame[frame.Tags.str.contains(tag)]
    return frame.groupby(['Date']).agg({'Tags': 'count'}).reset_index()


df_zara_filter = filter_by_year(year_input)

month_select = sorted(df_zara_filter['Month'].unique())

month_input = st.sidebar.multiselect('Mes', month_select)

df_zara_filter: pd.DataFrame = filter_by_month(year_input, month_input)

st.subheader('Dataframe Filtrado')
st.dataframe(df_zara_filter)

df_zara_filter_group = group_by_date(df_zara_filter)

st.subheader('Cantidad de Registros')
st.write('En la siguiente gráfica se puede observar la cantidad de datos obtenidos por día')
st.write(f'En total la cantidad es: {df_zara_filter_group.comment.sum()}')
st.write('Registros Obtenidos vs Día')

fig = px.line(
    x=df_zara_filter_group['Date'],
    y=df_zara_filter_group['comment'],
    labels={'x': 'Día', 'y': 'Cantidad'}
)

st.plotly_chart(fig, use_container_width=True)

# Tags

text_integrate = ' '.join(df_zara_filter.dropna().Tags.values)
words = list(set(text_integrate.split(' ')))
tags = list(filter(lambda x: len(x) > 1, words))

words_count = []

for i in tags:
    count = text_integrate.count(i)
    words_count.append([i, count])

df_words = pd.DataFrame(words_count, columns=['Word', 'Count'])
df_words = df_words.sort_values(by='Count')

st.subheader('Dataframe de las Etiquetas')
st.dataframe(df_words)

st.subheader('Principales Etiquetas')
st.write('Observamos las principales Etiquetas - Hashtags vs Cantidad')

fig = px.bar(
    y=df_words[-15:]['Word'],
    x=df_words[-15:]['Count'],
    labels={'y': 'Hastag', 'x': 'Cantidad'},
    orientation='h'
)

st.plotly_chart(fig, use_container_width=True)

tag_input = st.sidebar.selectbox('Seleccionar Etiqueta', df_words['Word'][-10:])

df_tag = filter_by_tags(df_zara_filter, tag_input)

st.subheader('Evolución de la Etiqueta')
st.write('Podemos observar como ha sido la evolución de la etiqueta en el mes - Etiqueta vs Día')

fig_tag = px.line(
    x=df_tag['Date'],
    y=df_tag['Tags'],
    labels={'x': 'Día', 'y': 'Cantidad'}
)

st.plotly_chart(fig_tag, use_container_width=True)

df_zara_filter_text = df_zara_filter['comment'].dropna().drop_duplicates()
text = ' '.join(df_zara_filter_text.values)
wc = WordCloud(background_color="white")
wc.generate(text)

st.subheader('Palabras más concurrentes')
st.write('Como podemos observar como Racismo o Racista que tiene un fuerte impacto en los comentarios')

fig, ax = plt.subplots()
ax.imshow(wc, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)


st.title('Análisis de Sentimiento de H&M')

st.subheader('H&M Análisis de Sentimientos')
df_hm_pred_sentimental = load_file('hm_pnl_pred.csv', 'csv')
df_hm_perc_sentimental = load_file('hm_pnl_percent.csv', 'csv')

row1Senti, row2Senti = st.columns(2)

with row1Senti:
    st.dataframe(df_hm_pred_sentimental)

with row2Senti:
    st.dataframe(df_hm_perc_sentimental)