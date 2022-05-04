# 2nd step: Process data from artefact/data.csv

import re
import time

# Pipeline
import mlflow

# Data
import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# Preprocessing
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def repartition_visualisation_graph(data, path, config):
    """This function create a graph of the repartition of the data.

    Args:
        data (Dataframe): dataframe with the preprocessed data
        path (str): path to save the graph
        config (dict): configuration options
    """
    # Plotting the univariate distribution of the data
    column = config["y_classificaton_column"]

    fig, ax = plt.subplots()
    fig.suptitle("Repartitions of " + column + "s", fontsize=12)
    data[column].reset_index().groupby(column).count().sort_values(by="index")\
    .plot(kind="barh", legend=False, ax=ax).grid(axis='x')
    fig.savefig(path)
    plt.close(fig)

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
    # # Take 10% of the data and profile report
    rows = data.sample(frac=0.10)
    profile = ProfileReport(rows, title="Profile report", progress_bar=False,
                            vars={"num": {"low_categorical_threshold": 0}})
    profile.to_file("artefact/panda_report_output.html")
    mlflow.log_artifact("artefact/panda_report_output.html")


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
    # Preprocessing Textual data

    stopwords = nltk.corpus.stopwords.words("english")
    data.loc[:, "Clean_Corpus"] = data.loc[:, "Corpus"].apply(
        utils_preprocessing_corpus, args=(stopwords, config["stemming"], config["lemmitization"]))

    ######################################################################################
    end = time.time()
    print(f"Preprocessing done in {end - start} seconds")
    save_preprocess_data(data)
    repartition_visualisation(data, config)
    return data
