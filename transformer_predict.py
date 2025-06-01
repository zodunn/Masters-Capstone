"""
music21: A Toolkit for Computer-Aided Musical Analysis and Computational Musicology
Author: Michael Scott Asato Cuthbert
Github: https://github.com/cuthbertLab/music21
Main page: https://www.music21.org/music21docs/
"""
import music21
"""
numpy: The fundamental package for scientific computing with Python.
Github: https://github.com/numpy/numpy
Main page: https://numpy.org/
"""
import numpy as np
"""
os: internal python module
docs: https://docs.python.org/3/library/os.html
"""
from os import listdir
"""
keras: a deep learning API
Github: https://github.com/keras-team/keras
Main page: https://keras.io/
"""
import keras
"""
keras module for creating model layers
"""
from keras import layers
"""
keras module for creating Model object
"""
from keras import Model
"""
sklearn: machine learning package for python
Github: https://github.com/scikit-learn/scikit-learn
Main page: https://scikit-learn.org/stable/
Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds
"""
from sklearn.model_selection import KFold
"""
pretty midi: Utility functions for handling MIDI data in a nice/intuitive way.
Authors: Colin Raffel and Daniel P. W. Ellis
Github: https://github.com/craffel/pretty-midi
Main page: https://craffel.github.io/pretty-midi/
"""
import pretty_midi


def import_music():
    """
    Reads in music xml files using music21 and extracts pitch, duration, offset, and fingering for each song and saves the data into an array
    :return: array of songs, each of which is an array of notes and their features
    """
    book2_path = 'music/book_2/xmls/'
    book_files = [(f, book2_path + f) for f in listdir(book2_path)]
    book3_path = 'music/book_3/xmls/'
    book_files = book_files + [(f, book3_path + f) for f in listdir(book3_path)]
    # print(book_files)

    songs = []
    for file_name, file_path in book_files:
        notes = []
        # Load the MusicXML file
        score = music21.converter.parse(file_path)

        # Extract note information
        for element in score.recurse().notes:
            fingering = -1
            if len(element.articulations) > 0:
                for articulation in element.articulations:
                    if isinstance(articulation, music21.articulations.Fingering):
                        fingering = articulation.fingerNumber
            notes.append([
                pitch_to_midi(element.pitch.nameWithOctave),
                float(element.duration.quarterLength),
                float(element.offset),
                fingering
            ])
        songs.append(notes)
    return songs


def pitch_to_midi(pitch):
    """
    Converts a string pitch into a midi note using pretty midi
    :param pitch: the pitch to convert
    :return: an integer representing the pitch
    """
    pitch_to_convert = pitch.replace('-', '!')
    try:
        return pretty_midi.note_name_to_number(pitch_to_convert)
    except ValueError:
        # Handle unknown values
        return 0


def transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    """
    Creates a set of layers that function as the transformer encoder for the model
    :param inputs: the previous layer of the model which is the sequence of notes that we want to have self attention
    :param head_size: dimension of each attention head
    :param num_heads: the number of heads in the transformer encoder
    :param ff_dim: the dimension of the feed forward part of the transformer
    :param dropout: the amount of dropout to apply
    :return: the transformer layers
    """
    # create self attention, dropout, residual connections, and normalization
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([x, inputs])
    x = layers.LayerNormalization()(x)

    # create feed forward layers and add more dropout
    x_ff = layers.Dense(ff_dim, activation="relu")(x)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)
    x_ff = layers.Dropout(dropout)(x_ff)

    # add another residual connection and normalization
    x = layers.Add()([x, x_ff])
    x = layers.LayerNormalization()(x)

    return x


def create_model(seq_length, pitch_vocab_size, embedding_dim):
    """
    Create transformer neural network model
    :param seq_length: dimension of note input
    :param pitch_vocab_size: dimension of vocabulary for embedding
    :param embedding_dim: dimension of embedding output
    :return: the created model
    """
    # create input layers for each of the features
    pitch_input = layers.Input(shape=(seq_length,), name="pitch")
    duration_input = layers.Input(shape=(seq_length,), name="duration")
    offset_input = layers.Input(shape=(seq_length,), name="offset")

    # create embedding layer for pitch and add the other features to the embedded pitch
    pitch_embedding = layers.Embedding(input_dim=pitch_vocab_size, output_dim=embedding_dim)(pitch_input)
    duration_embedding = layers.Dense(embedding_dim)(keras.ops.expand_dims(duration_input, -1))
    offset_embedding = layers.Dense(embedding_dim)(keras.ops.expand_dims(offset_input, -1))

    # concatenate embeddings
    x = layers.Add()([pitch_embedding, duration_embedding, offset_embedding])

    # create 2 transformer encoder layers
    for _ in range(2):
        x = transformer_encoder(x)

    # create the output layer - classify fingering for each note in the window
    out = layers.Dense(5, activation="softmax", name="fingering_output")(x)

    # Build Model
    model = Model(inputs=[pitch_input, duration_input, offset_input], outputs=out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # model.summary()

    return model


def create_sliding_windows(data, seq_length):
    """
    Create arrays of window_size notes and targets that include all features
    :param data: the song data to split into windows
    :param seq_length: how many notes to split into windows
    :return: windows of notes and targets
    """
    windows = []
    labels = []
    # for each song create windows
    for song in data:
        # create windows along the whole length of the song
        for i in range(len(song) - seq_length + 1):
            # get window of notes
            window = song[i:i + seq_length]
            # get note features
            windows.append([note[:-1] for note in window])
            # get note fingering labels
            labels.append([note[-1] for note in window])
    return np.array(windows), np.array(labels)


def train_and_test(x_pitch, x_duration, x_offset, y, seq_length, pitch_vocab_size, embedding_dim):
    """
    Do k-fold cross validation on dataset, each fold will create a new model and train/test it on portion of the data
    Also reports accuracy for each fold and averages accuracies of all folds together at the end
    :param x_pitch: training data for pitch feature
    :param x_duration: training data for duration feature
    :param x_offset: training data for offset feature
    :param y: labels for training data
    :param seq_length: length of note windows
    :param pitch_vocab_size: size of embedding vocabulary
    :param embedding_dim: size of embedding output
    :return: None
    """
    # create indices for folds
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    fold = 1
    metrics = []

    # for each set of indices train/test the model
    for train_idx, test_idx in kf.split(x_pitch):
        print(f"\nFold {fold}:")

        # get training data from kf indices
        x_train_pitch = x_pitch[train_idx]
        x_train_duration = x_duration[train_idx]
        x_train_offset = x_offset[train_idx]
        y_train = y[train_idx]

        # get testing data from kf indices
        x_test_pitch = x_pitch[test_idx]
        x_test_duration = x_duration[test_idx]
        x_test_offset = x_offset[test_idx]
        y_test = y[test_idx]

        # create model and train
        model = create_model(seq_length, pitch_vocab_size, embedding_dim)
        model.fit([x_train_pitch, x_train_duration, x_train_offset], y_train, epochs=200, batch_size=32, verbose=0)

        # evaluate model
        loss, accuracy = model.evaluate([x_test_pitch, x_test_duration, x_test_offset], y_test)
        metrics.append(accuracy)

        print(f"Loss: {loss}, Accuracy: {accuracy}")

        fold += 1

    avg = sum(metrics) / len(metrics)
    print(f"Average accuracy: {avg}")


def main():
    """
    Main driver method that imports the music data and passes it to the ML method
    :return: None
    """
    # Set up model inputs:
    # Sliding window size
    seq_length = 5
    # MIDI pitch range
    pitch_vocab_size = 128
    # Embedding output size
    embedding_dim = 32

    # Each sublist is a song, and each note has (pitch, duration, offset, fingering)
    data = import_music()

    # Create windows then separate out features
    x, y = create_sliding_windows(data, seq_length)
    x_pitch = x[:, :, 0]
    x_duration = x[:, :, 1]
    x_offset = x[:, :, 2]

    train_and_test(x_pitch, x_duration, x_offset, y, seq_length, pitch_vocab_size, embedding_dim)


if __name__ == '__main__':
    main()