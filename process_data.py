# 2nd step: Process data from artefact/data.csv

import re
import time

# Pipeline
import mlflow

# Data
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from umls_api.mysql_connection import db
from umls_api.semantic_network_queries import SemanticNetworkQueries
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Preprocessing
import nltk
from umls_api.column_type import ColumnType

from umls_api.metathesaurus_queries import get_tuple_of_languages_sources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def save_the_most_frequent_words(data, column, config):
    """This function save the most frequent words for each different column.

    Args:
        data (Dataframe): dataframe with the preprocessed data
        column (str): column to save the most frequent words
        config (dict): configuration options
    """
    if config["verbose"]:
        print(f"Saving the most frequent words [{column}]...")
    class_col = config["y_classificaton_column"]
    # Get only "Clean_Corpus" which has in "y_classificaton_column" the value of column var
    pd = data[data[class_col] == column]
    wordcloud = WordCloud(background_color='white', max_words=5).generate(
        pd['Clean_Corpus'].str.cat(sep=' '))
    path = 'artefact/wordcloud-' + column + '.png'
    wordcloud.to_file(path)
    mlflow.log_artifact(path)
    if config["verbose"]:
        print(f"Saving done.")


def generate_all_wordclouds(data, config):
    """This function generate all the wordclouds.

    Args:
        data (Dataframe): dataframe with the preprocessed data
        config (dict): configuration options
    """
    # Save the most frequent words for each different column
    column = config["y_classificaton_column"]
    labels = data[column].unique()
    for label in labels:
        save_the_most_frequent_words(data, label, config)


def repartition_visualisation_graph(data, path, config):
    """This function create a graph of the repartition of the data.

    Args:
        data (Dataframe): dataframe with the preprocessed data
        path (str): path to save the graph
        config (dict): configuration options
    """
    if config["verbose"]:
        print("Repartition graph generation...")
    # Plotting the univariate distribution of the data
    column = config["y_classificaton_column"]

    fig, ax = plt.subplots()
    fig.suptitle("Repartitions of " + column + "s", fontsize=12)
    data[column].reset_index().groupby(column).count().sort_values(by="index")\
        .plot(kind="barh", legend=False, ax=ax).grid(axis='x')
    fig.savefig(path)
    plt.close(fig)
    mlflow.log_artifact(path)
    if config["verbose"]:
        print("Repartition graph saved")


def repartition_visualisation(data, config):
    """Save the visualisation of the repartition of the data.

    Args:
        data (pd.DataFrame): dataframe with the preprocessed data
        config (config): config
    """
    if data is None:
        print("Error: data is None")
        return
    repartition_visualisation_graph(data, "artefact/repartitions.png", config)


def save_preprocess_data(data):
    """This function save the preprocessed data.

    Args:
        data (pd.DataFrame): dataframe with the preprocessed data
    """
    if data is None:
        print("Error: data is None")
        return
    print("Saving preprocessed data...")
    start = time.time()
    data.to_csv('artefact/preprocessed_data.csv', index=False)
    end = time.time()
    print(f"Saving done in {end - start} seconds")


def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)


def utils_preprocessing_corpus(text, stopwords=None, stemming=False, lemmitization=False):
    """This function prepare the corpus for the model.
    It can remove stopwords, lemmatize or stemming.

    Args:
        text (str): corpus to be processed
        stopwords (List[str], optional): stopwords to remove. Defaults to None.
        stemming (bool, optional): process stemming. Defaults to False.
        lemmitization (bool, optional): process lemmitization. Defaults to False.
    """
    # Check if the text is NaN
    if text != text:
        return ""
    # Convert text to lowercase
    text = text.lower()
    # Remove HTML tags
    text = remove_tags(text)
    # Remove links
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9-\s]', '', text)
    # Split every word
    text = text.split()
    # Remove stopwords
    if stopwords:
        text = [word for word in text if word not in stopwords]
    # Lemmatisation (convert the word into root word)
    if lemmitization is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text]
    # Stemming (remove -ing, -ly, ...)
    if stemming is True:
        ps = nltk.stem.porter.PorterStemmer()
        text = [ps.stem(word) for word in text]
    # Join the list of words
    text = " ".join(text)
    return text


def one_hot_encoding_sabs(sabset, l_sab):
    sabs = l_sab.split("/")
    return np.array([1 if x in sabs else 0 for x in sabset])


def get_preprocessed_labels_count(data):
    """This function get the preprocessed labels.

    Args:
        data (pd.DataFrame): dataframe with the preprocessed data

    Returns:
        pd.DataFrame: dataframe with the preprocessed labels
    """
    labels = data["Labels"].values
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    labels_count_np = vectorizer.fit_transform(labels)
    labels_count_np = labels_count_np.toarray().tolist()
    return labels_count_np


def get_preprocessed_sab(data):
    """This function preprocess the SAB column.

    Args:
        data (pd.Dataframe): dataframe with the preprocessed data

    Returns:
        pd.Dataframe: dataframe with the preprocessed data
    """
    all_sources = get_tuple_of_languages_sources()
    sabset = [x for x in all_sources]
    # Create a one-hot encoding of the SAB column from sabset
    values = data["SAB"].apply(
        lambda x: one_hot_encoding_sabs(sabset, x)).values
    return values


def get_preprocessed_parents_types(data: any, col_type: ColumnType):
    """This function preprocess the Parents Types column.

    Args:
        data (pd.Dataframe): dataframe with the preprocessed data

    Returns:
        pd.Dataframe: dataframe with the preprocessed data
    """
    # Remove NaN values and replace empty string
    data['Parents_Types'] = data['Parents_Types'].fillna("")

    parents_types = data["Parents_Types"].values
    sty = SemanticNetworkQueries(db)
    if col_type == ColumnType.GUI:
        all_gui = sty.get_all_single_gui()
        type_voc = [i[0] for i in all_gui]
    else:
        all_tui = sty.get_all_tui()
        type_voc = [i[0] for i in all_tui]
    vectorizer = TfidfVectorizer(vocabulary=type_voc, lowercase=False)
    type_frequency = vectorizer.fit_transform(parents_types).toarray()
    return type_frequency


def preprocess(config):
    """This function preprocess the data.

    Args:
        config (any): configuration options

    Returns:
        data (pd.DataFrame): dataframe with the preprocessed data
    """
    if config is None:
        print("Error: config is None")
        return None
    print("Preprocessing data...")
    start = time.time()
    # Import data.csv
    data = pd.read_csv('artefact/data.csv')

    # Log numbers of rows in MLflow
    nb_rows = len(data.index)
    mlflow.log_param("nb_rows", nb_rows)

    ######################################################################################
    # Preprocessing Corpus - Cleaning corpus

    stopwords = nltk.corpus.stopwords.words("english")
    data.loc[:, "Clean_Corpus"] = data.loc[:, "Corpus"].apply(
        utils_preprocessing_corpus, args=(stopwords, config["stemming"], config["lemmitization"]))
    # Put Prefered_Label in front of "Clean_Corpus" value
    data["Clean_Corpus"] = data["Prefered_Label"] + " " + data["Clean_Corpus"]

    ######################################################################################

    end = time.time()
    print(f"Preprocessing done in {end - start} seconds")
    # Drop "Corpus" column
    data = data.drop(columns=["Corpus", "Prefered_Label"])
    save_preprocess_data(data)
    repartition_visualisation(data, config)
    generate_all_wordclouds(data, config)
    return data
