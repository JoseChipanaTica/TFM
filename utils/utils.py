import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from collections import defaultdict
from nltk.corpus import stopwords

plt.tight_layout()
plt.style.use('dark_background')

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

    # axes[0].set_title(f'Top {n} most common unigrams in less_toxic comments', fontsize=15)
    # axes[1].set_title(f'Top {n} most common unigrams in more_toxic comments', fontsize=15)

    return plt


def show_pca(df: pd.DataFrame):
    return px.scatter(x=df['PCA1'], y=df['PCA2'], color=df['topic'])
