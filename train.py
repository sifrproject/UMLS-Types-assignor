import time

# Data
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Word Embedding
import gensim

# Neural network
from tensorflow.keras import models, layers, preprocessing as kprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics as sk_metrics
from keras import metrics
from keras.utils.vis_utils import plot_model  # Plot

import mlflow
import mlflow.keras

from process_data import get_preprocessed_labels_count, get_preprocessed_parents_types, get_preprocessed_sab, repartition_visualisation_graph
from umls_api.column_type import ColumnType


def get_processed_data(config):
    """Get preprocessed data stored in artefact/preprocessed_data.csv

    Args:
        config (dict): config

    Returns:
        DataFrame: data
    """
    data = pd.read_csv('artefact/preprocessed_data.csv')
    column = config["y_classificaton_column"]
    excluded = config["drop_classificaton_columns"]
    if excluded and len(excluded) > 0:
        if config["verbose"]:
            print("Excluding columns:", excluded)
        # Drop rows in data[column] that have one of the values in excluded
        data = data[~data[column].isin(excluded)]
    max_nb_data_per_class = config["max_nb_data_per_class"]
    if max_nb_data_per_class:
        if config["verbose"]:
            print("Max nb data per class:", max_nb_data_per_class)
        data = data.groupby(column).apply(lambda x: x.sample(max_nb_data_per_class)
                                          if len(x) > max_nb_data_per_class else x).reset_index(drop=True)
        repartition_visualisation_graph(
            data, "artefact/training-repartitions.png", config)
    if "Labels_Count" in config["attributes_features"]:
        if config["verbose"]:
            print("Get processed labels count...")
        data["Labels_Count"] = get_preprocessed_labels_count(data)
    data = data.drop(columns=["Labels"])
    return data


def get_nb_rows(list_of_features):
    """Get the number of rows of a list of features

    Args:
        list_of_features (List[List[Any]]): list of features

    Returns:
        int: nb_rows
    """
    return list_of_features[0].shape[0]


def get_train_test_data(data, config):
    """Get train and test data

    Args:
        data (Dataframe): data
        config (dict): config

    Returns:
        X_train_attributes, X_test_attributes, X_train_corpus, X_test_corpus, y_train, y_test

    """
    print("Splitting data...")
    start = time.time()
    df_train, df_test = train_test_split(data, test_size=config["test_size"])

    # Get values
    column = config["y_classificaton_column"]
    y_train = df_train[column].values
    y_test = df_test[column].values

    train_features = []
    test_features = []

    if "Has_Def" in config["attributes_features"]:
        if config["verbose"]:
            print("Get processed Has_Definition...")
        X_train_has_def = np.stack(df_train["Has_Definition"].values)
        X_test_has_def = np.stack(df_test["Has_Definition"].values)
        train_features.append(X_train_has_def)
        test_features.append(X_test_has_def)

    if "SAB" in config["attributes_features"]:
        if config["verbose"]:
            print("Get processed SAB...")
        train_sab = get_preprocessed_sab(df_train)
        X_train_sab = np.stack(train_sab)
        test_sab = get_preprocessed_sab(df_test)
        X_test_sab = np.stack(test_sab)
        train_features.append(X_train_sab)
        test_features.append(X_test_sab)

    if "Labels_Count" in config["attributes_features"]:
        if config["verbose"]:
            print("Get processed Labels_Count...")
        X_train_labels_count = np.stack(df_train["Labels_Count"].values)
        X_test_labels_count = np.stack(df_test["Labels_Count"].values)
        train_features.append(X_train_labels_count)
        test_features.append(X_test_labels_count)

    if "Parents_Types" in config["attributes_features"]:
        type = ColumnType.TUI if config["y_classificaton_column"] == "TUI" \
            else ColumnType.GUI
        X_train_parents_type = get_preprocessed_parents_types(df_train, type)
        X_test_parents_type = get_preprocessed_parents_types(df_test, type)
        train_features.append(X_train_parents_type)
        test_features.append(X_test_parents_type)

    if len(train_features) == 0:
        if config["verbose"]:
            print("No features to train on")
        X_train_has_def = np.stack(df_train["Has_Definition"].values)
        X_test_has_def = np.stack(df_test["Has_Definition"].values)
        train_features.append(X_train_has_def)
        test_features.append(X_test_has_def)
        print("Using default features: Has_Def")
    elif config["verbose"]:
        print("Using features:", config["attributes_features"])

    if config["verbose"]:
        print("Attributes shape")
        sum = 0
        for count, i in enumerate(train_features):
            try:
                if len(i.shape) > 1:
                    cols = i.shape[1]
                else:
                    cols = 1
            except:
                cols = 0
            sum += cols
            print(cols, end=" ")
            if count != len(train_features) - 1:
                print("+", end=" ")
        print("= ", sum)

    X_train_atrbts = [[]] * get_nb_rows(train_features)
    X_test_atrbts = [[]] * get_nb_rows(test_features)

    for i in train_features:
        if i is not None:
            X_train_atrbts = np.column_stack([X_train_atrbts, i])

    for i in test_features:
        if i is not None:
            X_test_atrbts = np.column_stack([X_test_atrbts, i])

    X_train_corpus = df_train["Clean_Corpus"].values
    X_test_corpus = df_test["Clean_Corpus"].values

    print("Splitting data done in %.2f seconds" % (time.time() - start))
    return X_train_atrbts, X_test_atrbts, X_train_corpus, X_test_corpus, y_train, y_test

######################################################################################
# Word Embedding


def create_list_unigrams(corpus):
    """Create list of lists of Unigrams (eg. ['I', 'am', 'a', 'student', '.'])

    Args:
        corpus (List[str]): corpus

    Returns:
        List: list of lists of Unigrams
    """
    lst_corpus = []
    for string in corpus:
        try:
            lst_words = string.split()
        except:
            lst_words = [""]
        lst_grams = [" ".join(lst_words[i:i+1])
                     for i in range(0, len(lst_words), 1)]
        if len(lst_grams) == 0:
            lst_grams = [""]
        lst_corpus.append(lst_grams)
    return lst_corpus


def train_w2v(train_corpus, config):
    """Train word2vec model

    Args:
        train_corpus (List[str]): Corpus that will be trained
        config (dict): config

    Returns:
        X_train_word_embedding, bigrams_detector, trigrams_detector, \
        tokenizer, nlp, dic_vocabulary
    """
    lst_corpus = create_list_unigrams(train_corpus)

    # Detect Bigrams (eg. ['I am', 'a student', '.'])
    bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=" ",
                                                     min_count=5, threshold=10)
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    # Detect Trigrams (eg. ['I am a', 'student.'])
    trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus],
                                                      delimiter=" ", min_count=5, threshold=10)
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

    # Fit Word2Vec
    nlp = gensim.models.word2vec.Word2Vec(lst_corpus,
                                          vector_size=config["vector_size"],
                                          window=config["window"], min_count=50, sg=1,
                                          epochs=config["w2v_epochs"])

    if config["verbose"]:
        print(
            f"The model has been trained on {nlp.corpus_total_words} words.\n")
        print("Shape of the word2vec model: ", nlp.wv.vectors.shape)
        print(
            f"It means that there are {nlp.wv.vectors.shape[0]} words in the model.")
        print(
            f"And each word has {nlp.wv.vectors.shape[1]} dimensions' vector.\n")

        word = "amine"
        try:
            print(
                f"The most similar words to '{word}' are:", nlp.wv.most_similar(word))
        except Exception as e:
            print(str(e))
            print(f"The word '{word}' is not in the vocabulary")

    # Tokenize text -> Indexation of each word (eg. {'NaN': 1, 'enzyme': 2, 'amine': 3...})
    tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN",
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(lst_corpus)
    dic_vocabulary = tokenizer.word_index

    # Create sequence -> Use index of each word from 'tokenizer' to create sentences
    # (eg. ['amine', 'enzyme'] -> [3, 2])
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

    # Padding sequence. Each sequence are composed of id or 0 to complete the size of 50
    # (eg. [3, 2] -> [3, 2, 0, 0, 0, 0, ..., 0])
    X_train_word_embedding = kprocessing.sequence.pad_sequences(
        lst_text2seq, maxlen=config["sequence_length"], padding="post", truncating="post")

    if config["verbose"]:
        print("Shape of the word embedding: ", X_train_word_embedding.shape)
        print(
            f"It means that there are {X_train_word_embedding.shape[0]} words in the model.")
        print(
            f"And each word has {X_train_word_embedding.shape[1]} dimensions' vector.\n")
        try:
            length_largest = len(max(train_corpus, key=len).split(' '))
            print("Highest length of definitions: ", length_largest)
        except Exception as e:
            print(str(e))
    return X_train_word_embedding, bigrams_detector, trigrams_detector, \
        tokenizer, nlp, dic_vocabulary


def apply_w2v(test_corpus: str, bigrams_detector, trigrams_detector, tokenizer, config):
    """Apply word2vec model to test corpus

    Args:
        test_corpus (List[str]): Corpus that will be tested
        bigrams_detector (Phraser): Bigrams detector
        trigrams_detector (Phraser): Trigrams detector
        tokenizer (any): tokenizer
        config (dict): config

    Returns:
        Any: X_test_word_embedding
    """
    # Create list of lists of Unigrams (eg. ['I', 'am', 'a', 'student', '.'])
    lst_corpus = create_list_unigrams(test_corpus)

    # Detect common bigrams and trigrams using the fitted detectors
    lst_corpus = list(bigrams_detector[lst_corpus])
    lst_corpus = list(trigrams_detector[lst_corpus])

    # Text to sequence with the fitted tokenizer
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

    # Padding sequence. Each sequence are composed of id or 0 to complete the size of 50
    X_test_word_embedding = kprocessing.sequence.pad_sequences(
        lst_text2seq, maxlen=config["sequence_length"], padding="post", truncating="post")
    return X_test_word_embedding


def word2vec(training_corpus, testing_corpus, config):
    """Train and apply word2vec model

    Args:
        training_corpus (List[str]): Corpus that will be trained
        testing_corpus (List[str]): Corpus that will be tested
        config (dict): config

    Returns:
       X_train_word_embedding, X_test_word_embedding, nlp, dic_vocabulary
    """
    print("Training word2vec model...")
    X_train_word_embedding, bigrams_detector, trigrams_detector, tokenizer, nlp, dic_vocabulary = \
        train_w2v(training_corpus, config)
    print("Word2vec model trained.")
    print("Applying word2vec model to testing_corpus...")
    X_test_word_embedding = apply_w2v(
        testing_corpus, bigrams_detector, trigrams_detector, tokenizer, config)
    return X_train_word_embedding, X_test_word_embedding, nlp, dic_vocabulary

######################################################################################


######################################################################################
# Modeling

def get_embeddings(dic_vocabulary, nlp, config):
    """Get embeddings

    Args:
        dic_vocabulary (dict): Dictionary of vocabulary
        nlp (Word2Vec): Word2Vec model
        config (dict): config

    Returns:
        NDArray[float64]: embeddings
    """
    # Start the matrix (length of vocabulary x vector size) with all 0s
    embeddings = np.zeros((len(dic_vocabulary)+1, config["vector_size"]))
    for word, idx in dic_vocabulary.items():
        # Update the row with vector
        try:
            embeddings[idx] = nlp[word]
        # If word not in model then skip and the row stays all 0s
        except:
            pass

    if config["verbose"]:
        word = "action"
        print("dic[word]:", dic_vocabulary[word], "|idx")
        print("embeddings[idx]:", embeddings[dic_vocabulary[word]].shape,
              "|vector")
    return embeddings


def get_input_layer_and_next_steps(type_model, config):
    """Get input layer and next steps

    Args:
        type_model (str): Type of model
        config (dict): config

    Returns:
        Inputs, Steps
    """
    steps = config["neural_network"][type_model]["steps"].copy()
    # Search for the first step with the type "Input"
    first_step = next(step for step in steps if step["type"] == "Input")
    if type_model == "multi_layer_perception":
        first_step["input_shape"] = config["numerical_data_shape"]
    inputs = layers.Input(
        shape=first_step["input_shape"], name=first_step["name"])
    # Remove first step from the list
    steps.remove(first_step)
    return inputs, steps


def create_model(type_model, config, embeddings=None, inputs=None):
    """Create model

    Args:
        type_model (str): Type of model
        config (dict): config
        embeddings (any, optional): Embeddings. Defaults to None.
        inputs (any, optional): Inputs. Defaults to None.

    Returns:
        Any: model
    """
    inputs, steps = get_input_layer_and_next_steps(type_model, config)

    x = inputs
    for i in steps:
        if i["type"] == "Embedding":
            x = layers.Embedding(input_dim=embeddings.shape[0],
                                 output_dim=embeddings.shape[1],
                                 weights=[embeddings],
                                 trainable=i["trainable"],
                                 name=i["name"])(x)
        elif i["type"] == "LSTM":
            x = layers.Bidirectional(layers.LSTM(
                units=i["units"], return_sequences=i["return_sequences"],
                dropout=i["dropout"]), name=i["name"])(x)
        elif i["type"] == "Dense":
            x = layers.Dense(units=i["units"], activation=i["activation"],
                             name=i["name"])(x)

    model = models.Model(inputs, x)
    return model


def create_word_embedding(config, embeddings):
    """Create word embedding

    Args:
        config (dict): config
        embeddings (any): embeddings

    Returns:
        any: model
    """
    return create_model("word_embedding", config, embeddings)


def create_multi_layer_perception(config):
    """Create multi layer perception

    Args:
        config (dict): config

    Returns:
        any: model
    """
    return create_model("multi_layer_perception", config)


def concatenate_neural_network(word_embedding, mlp, max_class, config):
    """Concatenate neural network

    Args:
        word_embedding (any): Word embedding model
        mlp (any): Multi layer perception model
        max_class (int): Max class
        config (dict): config

    Returns:
        Any: model
    """
    x = layers.concatenate([mlp.output, word_embedding.output])

    steps = config["neural_network"]["concatenate"]["steps"].copy()
    for i in steps:
        if i["type"] == "Dense":
            x = layers.Dense(units=i["units"], activation=i["activation"],
                             name=i["name"])(x)

    # Final layer
    final_activation = config["neural_network"]["out"]["activation"]
    name = config["neural_network"]["out"]["name"]
    y_out = layers.Dense(
        units=max_class, activation=final_activation, name=name)(x)

    return y_out


def get_model(nlp, dic_vocabulary, max_class, config):
    """Get model

    Args:
        nlp (Word2Vec): Word2Vec model
        dic_vocabulary (dict): Dictionary of vocabulary
        max_class (int): Max class
        config (dict): config

    Returns:
        Any: Final model
    """
    embeddings = get_embeddings(dic_vocabulary, nlp, config)

    word_embedding = create_word_embedding(config, embeddings)
    mlp = create_multi_layer_perception(config)
    y_out = concatenate_neural_network(word_embedding, mlp, max_class, config)

    optimize = config["neural_network"]["optimizer"]
    # metrics = config["neural_network"]["metrics"]
    loss = config["neural_network"]["loss"]
    model_mixed_data = models.Model(
        inputs=[word_embedding.input, mlp.input], outputs=y_out)
    model_mixed_data.compile(
        loss=loss, optimizer=optimize["name"], metrics=[metrics.binary_accuracy, "accuracy"])

    plot_model(model_mixed_data, to_file='artefact/model_plot.png',
               show_shapes=True, show_layer_names=True)
    return model_mixed_data



def plot_results(training):
    """Plot results (loss and acc) and save it to artefact/training.png

    Args:
        training (any): Training history
    """
    # Summarize history for accuracy
    plt.plot(training.history['accuracy'])
    plt.plot(training.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('artefact/history_accuracy.png')
    plt.close()
    # Summarize history for loss
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('artefact/history_loss.png')
    plt.close()


def train_model(X_train_attributes, X_train_word_embedding, y_train, max_class, nlp,
                dic_vocabulary, config):
    """Train model

    Args:
        X_train_attributes ([any]): Attributes
        X_train_word_embedding ([any]): Word embedding
        y_train ([any]): Labels
        max_class (int): Max class
        nlp (Word2Vec): Word2Vec model
        dic_vocabulary (dict): Dictionary of vocabulary
        config (dict): config

    Returns:
        tuple: model, training
    """
    if config["verbose"]:
        print("Generating model...")
    model = get_model(nlp, dic_vocabulary, max_class, config)

    # Encode y_train
    dic_y_mapping = {n: label for n, label in
                     enumerate(np.unique(y_train))}
    inverse_dic = {v: k for k, v in dic_y_mapping.items()}
    y_train = np.array([inverse_dic[y] for y in y_train])

    # Train
    epochs = config["neural_network"]["epochs"]
    batch_size = config["neural_network"]["batch_size"]
    shuffle = config["neural_network"]["shuffle"]
    verbose = config["verbose"]
    if config["verbose"]:
        print("Training / Fitting model...")
    training = model.fit(x=[X_train_word_embedding, X_train_attributes], y=y_train,
                         batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose,
                         validation_split=config["test_size"])
    if config["verbose"]:
        plot_results(training)
    return model, training

######################################################################################

######################################################################################
# Evaluation of the model


def evaluate_multi_classif(model, history, y_test, predicted, config):
    """Evaluate multi classification

    Args:
        model (Model): Model
        history (any): History
        y_test ([any]): Labels
        predicted ([str]): Predicted labels
        config (dict): config
    """
    ## Accuracy, Precision, Recall
    accuracy = sk_metrics.accuracy_score(y_test, predicted)
    print("Accuracy:",  round(accuracy, 2))
    print("Detail:")
    clf_report = sk_metrics.classification_report(
        y_test, predicted, output_dict=True)
    plt.clf()
    if config["y_classificaton_column"] == "TUI":
        fig = plt.subplots(figsize=(25, 25))
        res_plot = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, linewidths=.2)
    else:
        res_plot = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    fig = res_plot.get_figure()
    fig.savefig("artefact/results.png")
    plt.clf()

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("max_data", config["max_nb_data_per_class"])
    
    # Log params
    mlflow.log_param("class", config["y_classificaton_column"])
    mlflow.log_param("features", " ".join(config["attributes_features"]))
    mlflow.log_param("excluded", " ".join(config["drop_classificaton_columns"]))

    # log artifacts
    mlflow.log_artifact("artefact/model_plot.png")
    mlflow.log_artifact("artefact/history_accuracy.png")
    mlflow.log_artifact("artefact/history_loss.png")
    mlflow.log_artifact("artefact/results.png")

    # log model
    mlflow.keras.log_model(model, "model")
    mlflow.end_run()


def test_model(model, history, X_test_attributes, X_test_word_embedding, y_train, y_test, config):
    """Test model

    Args:
        model (Model): Model
        history (any): History
        X_test_attributes ([[float]]): Attributes
        X_test_word_embedding (any): Word embedding
        y_train ([any]): Train Labels
        y_test ([any]): Test Labels
        config (dict): config
    """
    predicted_prob = model.predict([X_test_word_embedding, X_test_attributes])
    dic_y_mapping = {n: label for n, label in
                     enumerate(np.unique(y_train))}
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
    evaluate_multi_classif(model, history, y_test, predicted, config)

######################################################################################


def train_and_test(config):
    """Training and testing step

    Args:
        config (dict): config
    """
    if config["verbose"]:
        print("Loading training / testing data...")
    # Data preparation
    data = get_processed_data(config)
    X_train_attributes, X_test_attributes, X_train_corpus, X_test_corpus, y_train, y_test = \
        get_train_test_data(data, config)
    max_class = data[config["y_classificaton_column"]].nunique()

    # Word2Vec
    if config["verbose"]:
        print("Loading word2vec model...")
    X_train_word_embedding, X_test_word_embedding, nlp, dic_vocabulary = word2vec(
        X_train_corpus, X_test_corpus, config)

    nb_attributes = X_train_attributes.shape[1]
    config["numerical_data_shape"] = nb_attributes

    # Set up panda dataframe with word embedding and attributes and y
    column = config["y_classificaton_column"]
    datafram = pd.DataFrame()
    datafram[column] = y_train
    repartition_visualisation_graph(datafram, "artefact/repartition_visualisation_graph.png",
                                    config)

    # Train the model
    if config["verbose"]:
        print("Training model...")
    model, history = train_model(X_train_attributes, X_train_word_embedding,
                                 y_train, max_class, nlp, dic_vocabulary, config)

    # Test the model
    if config["verbose"]:
        print("Testing model...")
    test_model(model, history, X_test_attributes,
               X_test_word_embedding, y_train, y_test, config)
