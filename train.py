import os
import shutil

# Data
import pandas as pd
import numpy as np

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


def get_processed_data():
    """Get preprocessed data stored in artefact/preprocessed_data.csv

    Returns:
        DataFrame: data
    """
    return pd.read_csv('artefact/preprocessed_data.csv')


def get_train_test_data(data, config):
    """Get train and test data

    Args:
        data (Dataframe): data
        config (dict): config

    Returns:
        X_train_attributes, X_test_attributes, X_train_corpus, X_test_corpus, y_train, y_test

    """
    df_train, df_test = train_test_split(data, test_size=config["test_size"])

    # Get values
    column = config["y_classificaton_column"]
    y_train = df_train[column].values
    y_test = df_test[column].values

    X_train_attributes = df_train.loc[:, [
        'Nb_Parents', 'Nb_Children', 'Nb_Parents_Children_Known', 'Has_Definition']].values
    X_test_attributes = df_test.loc[:, [
        'Nb_Parents', 'Nb_Children', 'Nb_Parents_Children_Known', 'Has_Definition']].values

    X_train_corpus = df_train["Clean_Definition"].values
    X_test_corpus = df_test["Clean_Definition"].values

    return X_train_attributes, X_test_attributes, X_train_corpus, X_test_corpus, y_train, y_test

######################################################################################
# Word Embedding


def create_list_unigrams(corpus):
    """Create list of lists of Unigrams (eg. ['I', 'am', 'a', 'student', '.'])

    Args:
        corpus (str): corpus

    Returns:
        List: list of lists of Unigrams
    """
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [" ".join(lst_words[i:i+1])
                     for i in range(0, len(lst_words), 1)]
        if len(lst_grams) == 0:
            lst_grams = [""]
        lst_corpus.append(lst_grams)
    return lst_corpus


def train_w2v(train_corpus, config):
    """Train word2vec model

    Args:
        train_corpus (str): Corpus that will be trained
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
                                          window=config["window"], min_count=1, sg=1,
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
        except KeyError:
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
        length_largest = len(max(train_corpus, key=len).split(' '))
        print("Highest length of definitions: ", length_largest)
    return X_train_word_embedding, bigrams_detector, trigrams_detector, \
        tokenizer, nlp, dic_vocabulary


def apply_w2v(test_corpus: str, bigrams_detector, trigrams_detector, tokenizer, config):
    """Apply word2vec model to test corpus

    Args:
        test_corpus (str): Corpus that will be tested
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
        training_corpus (str): Corpus that will be trained
        testing_corpus (str): Corpus that will be tested
        config (dict): config

    Returns:
       X_train_word_embedding, X_test_word_embedding, nlp, dic_vocabulary
    """
    X_train_word_embedding, bigrams_detector, trigrams_detector, tokenizer, nlp, dic_vocabulary = \
        train_w2v(training_corpus, config)
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
        loss=loss, optimizer=optimize["name"], metrics=[metrics.binary_accuracy])

    # You need to install graphviz (see instructions at https://graphviz.gitlab.io/download/)
    plot_model(model_mixed_data, to_file='artefact/model_plot.png',
               show_shapes=True, show_layer_names=True)
    return model_mixed_data


def get_binary_loss(hist):
    """Get binary loss"""
    loss = hist.history['loss']
    loss_val = loss[len(loss) - 1]
    return loss_val


def get_binary_acc(hist):
    """Get binary acc"""
    acc = hist.history['binary_accuracy']
    acc_value = acc[len(acc) - 1]

    return acc_value


def get_validation_loss(hist):
    """Get validation loss"""
    val_loss = hist.history['val_loss']
    val_loss_value = val_loss[len(val_loss) - 1]

    return val_loss_value


def get_validation_acc(hist):
    """Get validation acc"""
    val_acc = hist.history['val_binary_accuracy']
    val_acc_value = val_acc[len(val_acc) - 1]

    return val_acc_value


def plot_results(training):
    """Plot results (loss and acc) and save it to artefact/training.png

    Args:
        training (any): Training history
    """
    # Plot loss and accuracy
    metrics = [k for k in training.history.keys() if (
        "loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    fig.savefig('artefact/training.png')
    plt.close(fig)


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
    res_plot = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    fig = res_plot.get_figure()
    fig.savefig("artefact/results.png")

    # Log all the MLflow config
    mlflow.log_param("nn_epoch", config["neural_network"]["epochs"])
    mlflow.log_param("nn_loss", config["neural_network"]["loss"])
    mlflow.log_param("nn_optimizer_name",
                     config["neural_network"]["optimizer"]["name"])
    mlflow.log_param("nn_metrics", metrics.categorical_accuracy)
    mlflow.log_param("w2c_vector_size", config["vector_size"])
    mlflow.log_param("w2c_window", config["window"])
    mlflow.log_param("w2c_epoch", config["w2v_epochs"])
    mlflow.log_param("w2c_sequence_length", config["sequence_length"])
    mlflow.log_dict(config, "config.yaml")

    # Calculate metrics
    binary_loss = get_binary_loss(history)
    binary_acc = get_binary_acc(history)
    validation_loss = get_validation_loss(history)
    validation_acc = get_validation_acc(history)

    # Log all the MLflow Metrics
    mlflow.log_metric("binary_loss", binary_loss)
    mlflow.log_metric("binary_acc", binary_acc)
    mlflow.log_metric("validation_loss", validation_loss)
    mlflow.log_metric("validation_acc", validation_acc)
    mlflow.log_metric("average_acc", accuracy)

    # log artifacts
    mlflow.log_artifact("artefact/model_plot.png")
    mlflow.log_artifact("artefact/training.png")
    mlflow.log_artifact("artefact/results.png")
    # Add output log to mlflow log if debug_output_path is file
    if config["debug_output_path"] and len(config["debug_output_path"]) > 0 \
            and os.path.isfile(config["debug_output_path"]):
        mlflow.log_artifact(config["debug_output_path"])

    # log model
    mlflow.keras.log_model(model, "model")
    try:
        shutil.rmtree("artefact/models")
        print("Directory artefact/models removed")
    except OSError as e:
        print("Error: %s : %s" % ("artefact/models", e.strerror))
    mlflow.keras.save_model(model, "artefact/models")

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

    # Data preparation
    data = get_processed_data()
    X_train_attributes, X_test_attributes, X_train_corpus, X_test_corpus, y_train, y_test = \
        get_train_test_data(data, config)
    max_class = data[config["y_classificaton_column"]].nunique()

    # Word2Vec
    X_train_word_embedding, X_test_word_embedding, nlp, dic_vocabulary = word2vec(
        X_train_corpus, X_test_corpus, config)

    # Train the model
    model, history = train_model(X_train_attributes, X_train_word_embedding,
                                 y_train, max_class, nlp, dic_vocabulary, config)

    # Test the model
    test_model(model, history, X_test_attributes,
               X_test_word_embedding, y_train, y_test, config)
