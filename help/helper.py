from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.datasets import fetch_20newsgroups

import numpy as np
import json
import pickle
import os

class Helper:
    dataset_20newsgroup_data_training = 'train'
    dataset_20newsgroup_data_testing = 'test'

    @staticmethod
    def hasFile(path):
        return os.path.exists(path)

    @staticmethod
    def stem(docs, use_tokenize=True):
        ps = SnowballStemmer('english')

        stemmed_docs = []
        for doc in docs:
            words = word_tokenize(doc.lower()) if use_tokenize else doc.split(' ')
            stemmed_docs.append(' '.join([ps.stem(word) for word in words]))

        return stemmed_docs

    @staticmethod
    def saveJsonTo(path ,data):
        with open(path, 'w') as file:
            json.dump(data, file)

    @staticmethod
    def loadJsonFrom(path):
        with open(path) as file:
            data = json.load(file)

        return data

    @staticmethod
    def saveObjectTo(path, object):
        with open(path, 'wb') as file:
            pickle.dump(object, file)

    @staticmethod
    def loadObjectFrom(path):
        with open(path, 'rb') as file:
            object = pickle.load(file)

        return object

    @staticmethod
    def printDictionary(dict, indent=0, keys_to_print=None):
        space = "  " * indent
        keys = (dict.keys() if keys_to_print is None else keys_to_print)
        rank_len = sorted(keys, key=lambda x: len(x), reverse=True)
        max_len = len(rank_len[0])
        for key in keys:
            if type(dict[key]) == type({}):
                print(space + str(key))
                Helper.printDictionary(dict[key], indent + 1)
            else:
                print(space + str(key) + (' ' * (max_len - len(key) + 1)) + ': ' + str(dict[key]))

    @staticmethod
    def buildAutoencoder(n_features, coded_unit, max_autoencoder_unit):
        from keras.models import Model
        from keras.layers import Input, Dense
        from keras.optimizers import Adam

        layers = []

        decoder_layers = []
        counter = 1
        while True:
            units = 2 ** counter
            if units > coded_unit and units < n_features and units <= max_autoencoder_unit:
                decoder_layers.append(units)
            if units >= n_features:
                break
            counter += 1

        encoder_layers = decoder_layers.copy()
        encoder_layers.reverse()

        layers.extend(encoder_layers.copy())
        layers.append(coded_unit)
        layers.extend(decoder_layers.copy())

        input = Input(shape=(n_features,))

        try:
            encoder = Dense(units=encoder_layers[0], activation='relu')(input)
            del encoder_layers[0]
            for n_unit in encoder_layers:
                encoder = Dense(units=n_unit, activation='relu')(encoder)
            del encoder_layers

            coded = Dense(units=coded_unit, activation='relu')(encoder)
        except:
            coded = Dense(units=coded_unit, activation='relu')(input)

        try:
            decoder = Dense(units=decoder_layers[0], activation='relu')(coded)
            del decoder_layers[0]
            for n_unit in decoder_layers:
                decoder = Dense(units=n_unit, activation='relu')(decoder)

            output = Dense(units=n_features, activation='sigmoid')(decoder)
        except:
            output = Dense(units=n_features, activation='sigmoid')(coded)

        autoencoder = Model(inputs=input, outputs=output)
        encoder = Model(inputs=input, outputs=coded)

        return layers ,encoder, autoencoder

