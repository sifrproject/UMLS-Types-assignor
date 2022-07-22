import multiprocessing
import time

# Data
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import nltk

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Word Embedding
import gensim
from graph_predictions import LinkedTree, Node, get_BIOPORTAL_API_KEY

# Neural network
from tensorflow.keras import models, layers, preprocessing as kprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics as sk_metrics
from keras import metrics
from keras.utils.vis_utils import plot_model  # Plot

import mlflow
import mlflow.keras

from process_data import apply_SAB_preprocess, get_preprocessed_parents_types, get_preprocessed_sab, repartition_visualisation_graph, utils_preprocessing_corpus
from umls_api.column_type import ColumnType
from umls_api.bioportal_api import BioPortalAPI


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
    type = ColumnType.TUI if config["y_classificaton_column"] == "TUI" \
            else ColumnType.GUI
    if type == ColumnType.TUI:
        data.drop(columns=["Parents_Types_GUI"], inplace=True)
        data.rename(columns={"Parents_Types_TUI": "Parents_Types"}, inplace=True)
    else:
        data.drop(columns=["Parents_Types_TUI"], inplace=True)
        data.rename(columns={"Parents_Types_GUI": "Parents_Types"}, inplace=True)
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
        X_train_attributes, X_test_attributes, X_train_corpus, X_test_corpus, 
        X_train_labels, X_test_labels, y_train, y_test

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

    X_train_labels = df_train["Labels"].values
    X_test_labels = df_test["Labels"].values

    print("Splitting data done in %.2f seconds" % (time.time() - start))
    return X_train_atrbts, X_test_atrbts, X_train_corpus, X_test_corpus, \
        X_train_labels, X_test_labels, y_train, y_test

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
                                                     min_count=30, progress_per=10000)
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    # Detect Trigrams (eg. ['I am a', 'student.'])
    trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus],
                                                      delimiter=" ", min_count=30, progress_per=10000)
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

    # https://www.datatechnotes.com/2019/05/word-embedding-with-keras-in-python.html
    # Fit Word2Vec
    cores = multiprocessing.cpu_count() # Count the number of cores in a computer
    nlp = gensim.models.word2vec.Word2Vec(lst_corpus,
                                          vector_size=config["vector_size"],
                                          window=config["window"], min_count=50, sg=1,
                                          epochs=config["w2v_epochs"],
                                          negative=20,
                                        workers=cores-1)

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
    return X_train_word_embedding, X_test_word_embedding, nlp, dic_vocabulary, \
        tokenizer, bigrams_detector, trigrams_detector


def bag_of_words_gen(X_train_labels, X_test_labels, config):
    """Train and apply bag of words model

    Args:
        X_train_labels (List[str]): Labels of training corpus
        X_test_labels (List[str]): Labels of testing corpus

    Returns:
       X_train_bow, X_test_bow, tokenizer
    """
    print("Training bag of words model...")
    tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN",
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(X_train_labels)

    train_sequences = tokenizer.texts_to_sequences(X_train_labels)
    X_train_bow = tokenizer.sequences_to_matrix(train_sequences, mode='tfidf')

    test_sequences = tokenizer.texts_to_sequences(X_test_labels)
    X_test_bow = tokenizer.sequences_to_matrix(test_sequences, mode='tfidf')

    return X_train_bow, X_test_bow, tokenizer

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
        elif i["type"] == "Dropout":
            x = layers.Dropout(i["rate"], name=i["name"])(x)
        elif i["type"] == "BatchNormalization":
            x = layers.BatchNormalization(name=i["name"])(x)
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


def create_bag_of_words(config):
    """Create bag of words

    Args:
        config (dict): config

    Returns:
        any: model
    """
    return create_model("bag_of_words", config)


def concatenate_neural_network(models, max_class, config):
    """Concatenate neural network

    Args:
        word_embedding (any): Word embedding model
        mlp (any): Multi layer perception model
        max_class (int): Max class
        config (dict): config

    Returns:
        Any: model
    """
    x = layers.concatenate(
        [i.output for i in models])

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
    models_list = []
    if "Def" in config["attributes_features"]:
        embeddings = get_embeddings(dic_vocabulary, nlp, config)
        word_embedding = create_word_embedding(config, embeddings)
        models_list.append(word_embedding)

    if "Has_Def" in config["attributes_features"] or "SAB" in config["attributes_features"] or \
            "Parents_Types" in config["attributes_features"]:
        mlp = create_multi_layer_perception(config)
        models_list.append(mlp)

    if "Labels" in config["attributes_features"]:
        bag_of_words = create_bag_of_words(config)
        models_list.append(bag_of_words)

    y_out = concatenate_neural_network(
        models_list, max_class, config)

    optimize = config["neural_network"]["optimizer"]
    # metrics = config["neural_network"]["metrics"]
    loss = config["neural_network"]["loss"]
    inputs = [i.input for i in models_list]
    model_mixed_data = models.Model(
        inputs=inputs, outputs=y_out)
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


def train_model(X_train_attributes, X_train_word_embedding, X_train_bow, y_train, max_class, nlp,
                dic_vocabulary, config):
    """Train model

    Args:
        X_train_attributes ([any]): Attributes
        X_train_word_embedding ([any]): Word embedding
        X_train_bow([any]): Bag of words
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

    features = []
    if "Def" in config["attributes_features"]:
        features.append(X_train_word_embedding)
    if "Has_Def" in config["attributes_features"] or "SAB" in config["attributes_features"] or \
            "Parents_Types" in config["attributes_features"]:
        features.append(X_train_attributes)
    if "Labels" in config["attributes_features"]:
        features.append(X_train_bow)

    training = model.fit(x=features, y=y_train,
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
        res_plot = sns.heatmap(pd.DataFrame(
            clf_report).iloc[:-1, :].T, annot=True, linewidths=.2)
    else:
        res_plot = sns.heatmap(pd.DataFrame(
            clf_report).iloc[:-1, :].T, annot=True)
    fig = res_plot.get_figure()
    fig.savefig("artefact/results.png")
    plt.clf()

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("max_data", config["max_nb_data_per_class"])

    # Log params
    mlflow.log_param("class", config["y_classificaton_column"])
    mlflow.log_param("features", " ".join(config["attributes_features"]))
    mlflow.log_param("excluded", " ".join(
        config["drop_classificaton_columns"]))

    # log artifacts
    mlflow.log_artifact("artefact/model_plot.png")
    mlflow.log_artifact("artefact/history_accuracy.png")
    mlflow.log_artifact("artefact/history_loss.png")
    mlflow.log_artifact("artefact/results.png")

    # log model
    mlflow.keras.log_model(model, "model")
    mlflow.end_run()


def test_model(model, history, X_test_attributes, X_test_word_embedding, X_test_bow,
               y_train, y_test, config):
    """Test model

    Args:
        model (Model): Model
        history (any): History
        X_test_attributes ([[float]]): Attributes
        X_test_word_embedding (any): Word embedding
        X_test_bow ([any]): Bag of words
        y_train ([any]): Train Labels
        y_test ([any]): Test Labels
        config (dict): config
    """
    features = []
    if "Def" in config["attributes_features"]:
        features.append(X_test_word_embedding)
    if "Has_Def" in config["attributes_features"] or "SAB" in config["attributes_features"] or \
            "Parents_Types" in config["attributes_features"]:
        features.append(X_test_attributes)
    if "Labels" in config["attributes_features"]:
        features.append(X_test_bow)
    predicted_prob = model.predict(features)
    dic_y_mapping = {n: label for n, label in
                     enumerate(np.unique(y_train))}
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
    evaluate_multi_classif(model, history, y_test, predicted, config)
    return dic_y_mapping

######################################################################################

def set_graph_prediction(source, bow_tokenizer, we_tokenizer, we_bigrams_detector, we_trigrams_detector, config):

    BIOPORTAL_API_KEY = get_BIOPORTAL_API_KEY()
    if BIOPORTAL_API_KEY is None:
        print("BIOPORTAL_API_KEY is not set.")
        return None, None
    portal = BioPortalAPI(api_key=BIOPORTAL_API_KEY)

    print("Loading concepts...")
    roots = portal.get_roots_of_tree(source)
    
    stopwords = nltk.corpus.stopwords.words("english")

    root_node_elements = {
        "pref_label": 'root',
        "labels": '',
        "source": '',
        "code_id": "https://data.bioontology.org/ontologies/" + str(source) + "/classes/roots",
        "has_definition": False,
        "definition": '',
        "parents_type": None,
        "parents_code_id": "",
    }
    root_node = Node(root_node_elements)
    linked_tree = LinkedTree(root_node)
    children_links_list = [ node['links']['self'] for node in roots ]

    def recursive_add_all_nodes(portal, children_links_list, parents_code_id):
        for link in children_links_list:
            features = portal.get_features_from_link(link, parents_code_id)
            if features is None:
                continue
            
            # Labels
            if bow_tokenizer is not None:
                sequences = bow_tokenizer.texts_to_sequences([features['labels']])
                bow = bow_tokenizer.sequences_to_matrix(sequences, mode='tfidf')
                features['labels'] = bow[0]
            # SAB
            features['source'] = apply_SAB_preprocess(features['source'])
            # Corpus
            definition = utils_preprocessing_corpus(features['definition'], stopwords, config["stemming"], config["lemmitization"])
            we_definitions = apply_w2v([definition], we_bigrams_detector, we_trigrams_detector, we_tokenizer, config)
            features['definition'] = we_definitions[0]
            
            new_node = Node(features)
            linked_tree.add_node(parents_code_id, new_node)
            children_link = features['children']
            new_parents_code_id = features['code_id']
            children_links_list = portal.get_children_links(children_link)
            recursive_add_all_nodes(
                portal, children_links_list, new_parents_code_id)
        return

    print("Loading concepts...")
    recursive_add_all_nodes(portal, children_links_list, root_node.code_id)
    return linked_tree, source

def test_graph(model, linked_tree, dic_y_mapping, source, config):
    print("Predicting graph...")
    linked_tree.predict_graph(model, dic_y_mapping, config)
    linked_tree.save_prediction_to_ttl(source, config)
    return linked_tree

def train_and_test(config):
    """Training and testing step

    Args:
        config (dict): config
    """
    if config["verbose"]:
        print("Loading training / testing data...")

    # Data preparation

    data = get_processed_data(config)
    X_train_attributes, X_test_attributes, X_train_corpus, X_test_corpus, X_train_labels, \
        X_test_labels, y_train, y_test = get_train_test_data(data, config)
    max_class = data[config["y_classificaton_column"]].nunique()

    # Update MLP input shape
    nb_attributes = X_train_attributes.shape[1]
    config["neural_network"]["multi_layer_perception"]["steps"][0]["input_shape"] = nb_attributes

    # Word2Vec
    if config["verbose"]:
        print("Loading word2vec model...")
    X_train_word_embedding, X_test_word_embedding, nlp, dic_vocabulary, \
        we_tokenizer, we_bigrams_detector, we_trigrams_detector = \
            word2vec(X_train_corpus, X_test_corpus, config)

    # Bag of words
    if config["verbose"]:
        print("Loading bag of words model...")
    X_train_bow, X_test_bow, bow_tokenizer = bag_of_words_gen(
        X_train_labels, X_test_labels, config)
    config["neural_network"]["bag_of_words"]["steps"][0]["input_shape"] = X_test_bow.shape[1]

    # Debug
    column = config["y_classificaton_column"]
    datafram = pd.DataFrame()
    datafram[column] = y_train
    repartition_visualisation_graph(datafram, "artefact/training2-repartition.png",
                                    config)

    # Train the model
    if config["verbose"]:
        print("Training model...")
    model, history = train_model(X_train_attributes, X_train_word_embedding, X_train_bow,
                                 y_train, max_class, nlp, dic_vocabulary, config)

    # Test the model
    if config["verbose"]:
        print("Testing model...")
    dic_y_mapping = test_model(model, history, X_test_attributes,
               X_test_word_embedding, X_test_bow, y_train, y_test, config)
    
    if config["test_source"] and len(config["test_source"]) > 0:
        sources = config["test_source"]
    else:
        return None
    for source in sources:
        linked_tree, source = set_graph_prediction(source, bow_tokenizer, we_tokenizer, we_bigrams_detector, we_trigrams_detector, config)
        if linked_tree is not None:
            test_graph(model, linked_tree, dic_y_mapping, source, config)
