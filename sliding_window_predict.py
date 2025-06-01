"""
music21: A Toolkit for Computer-Aided Musical Analysis and Computational Musicology
Author: Michael Scott Asato Cuthbert
Github: https://github.com/cuthbertLab/music21
Main page: https://www.music21.org/music21docs/
"""
import music21
"""
pandas: an open source data analysis and manipulation tool, built on top of the Python programming language.
Github: https://github.com/pandas-dev/pandas
Main page: https://pandas.pydata.org/
"""
import pandas as pd
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
sklearn: machine learning package for python
Github: https://github.com/scikit-learn/scikit-learn
Main page: https://scikit-learn.org/stable/
train_test_split is a module that separates training and testing data
"""
from sklearn.model_selection import train_test_split
"""
pretty midi: Utility functions for handling MIDI data in a nice/intuitive way.
Authors: Colin Raffel and Daniel P. W. Ellis
Github: https://github.com/craffel/pretty-midi
Main page: https://craffel.github.io/pretty-midi/
"""
import pretty_midi


def import_music(return_dataframe=False):
    """
    Reads in music xml files using music21 and extracts pitch, duration, offset, and fingering for each song and saves the data into a dataframe and song array
    :param return_dataframe: tells the method to return dataframe and songs or just songs array
    :return: dataframe of songs and features or an array of songs
    """
    book2_path = 'music/book_2/xmls/'
    book_files = [(f, book2_path + f) for f in listdir(book2_path)]
    book3_path = 'music/book_3/xmls/'
    book_files = book_files + [(f, book3_path + f) for f in listdir(book3_path)]
    # print(book_files)

    notes = []
    for file_name, file_path in book_files:
        # Load the MusicXML file
        score = music21.converter.parse(file_path)

        # Extract note information
        for element in score.recurse().notes:
            fingering = -1
            if len(element.articulations) > 0:
                for articulation in element.articulations:
                    if isinstance(articulation, music21.articulations.Fingering):
                        fingering = articulation.fingerNumber
            notes.append({
                'song': file_name,
                'pitch': pitch_to_midi(element.pitch.nameWithOctave),
                'duration': float(element.duration.quarterLength),
                'offset': float(element.offset),
                'fingering': fingering,
            })
    # Create DataFrame
    df = pd.DataFrame(notes)

    songs = []
    for file_name, file_path in book_files:
        song = {'song': file_name}
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
            notes.append({
                'pitch': pitch_to_midi(element.pitch.nameWithOctave),
                'duration': float(element.duration.quarterLength),
                'offset': float(element.offset),
                'fingering': fingering,
            })
        song['notes'] = notes
        songs.append(song)

    if return_dataframe:
        return df, songs
    else:
        return songs


def create_fake_data(df, num_songs):
    """
    Creates fake songs by randomly selecting pitches and duration then calculates the fingering based on probability distribution and offset based on previous sum of durations
    resetting every 4 'beats' or time steps
    :param df: the original song data to calculate pitches and fingerings from
    :param num_songs: how many songs to generate
    :return: array of fake songs
    """
    # get unique pitches
    unique_pitches = df['pitch'].unique()
    distributions = {}
    # calculate the distributions of fingerings for each pitch
    for pitch in unique_pitches:
        fingering_distribution = calculate_fingering_distribution(pitch, df)
        distributions[pitch] = fingering_distribution

    # get unique durations
    durations = df['duration'].unique()
    # remove durations of length 0
    durations = np.delete(durations, np.where(durations == 0.0))
    fake_songs = []
    offset = 0.0
    # create new songs
    for i in range(num_songs):
        song = {'song': 'fake_song_' + str(i)}
        song_notes = []
        for j in range(0, 300):
            note = np.random.choice(unique_pitches)
            duration = np.random.choice(durations)
            if duration + offset > 4:
                duration = 4 - offset
            chosen_fingering = choose_fingering(distributions[note])
            song_notes.append({
                'pitch': note,
                'duration': duration,
                'offset': offset,
                'fingering': chosen_fingering
            })
            offset += duration
            if offset >= 4:
                offset = 0.0
        song['notes'] = song_notes
        fake_songs.append(song)
    return fake_songs


def calculate_fingering_distribution(pitch, df):
    """
    Calculates fingering distribution based on probability distribution
    :param pitch: the pitch to calculate fingering distribution for
    :param df: the song data to count the pitches and fingerings from
    :return: fingering probability distribution for the pitch
    """
    # Filter DataFrame for the given pitch
    pitch_data = df[df['pitch'] == pitch]

    # Calculate fingering counts and probabilities
    fingering_counts = pitch_data['fingering'].value_counts(normalize=True)
    return fingering_counts


def choose_fingering(fingering_distribution):
    """
    Uses the fingering distributions to generate a fingering
    :param fingering_distribution: the fingering distribution of a pitch
    :return: an integer representing the chosen fingering
    """
    # Choose a fingering based on the probability distribution
    fingers = fingering_distribution.index
    probabilities = fingering_distribution.values
    return np.random.choice(fingers, p=probabilities)


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


def create_sliding_windows(notes, window_size=2, include_features=False):
    """
    Create arrays of window_size notes and targets that optionally includes all features
    :param notes: all the notes of all the songs
    :param window_size: size of the sliding windows (how many notes to put in each array)
    :param include_features: true/false for if we want all features or just pitch
    :return: windows of notes and targets
    """
    x, y = [], []
    # get windows with all features
    if include_features:
        for i in range(len(notes)-(2 * window_size)):
            # Take window of notes
            x.append([[note['pitch'], note['duration'], note['offset']] for note in notes[i:i + (2 * window_size + 1)]])
            # Fingering of the middle note
            y.append(notes[i+2]['fingering'])
        return np.array(x), np.array(y)
    # get windows with just pitch
    else:
        for i in range(len(notes)-(2 * window_size)):
            # Take window of notes
            x.append([note['pitch'] for note in notes[i:i + (2 * window_size + 1)]])
            # Fingering of the middle note
            y.append(notes[i+2]['fingering'])
        return np.array(x), np.array(y)


def prepare_dataset(songs, include_features=False):
    """
    Convert array of songs into windows of notes and targets
    :param songs: array of songs that contain notes
    :param include_features: true/false for if we want all features or just pitch
    :return: arrays of arrays of notes and targets
    """
    x, y = [], []
    # for each song create windows of notes and targets
    for song in songs:
        x_song, y_song = create_sliding_windows(song['notes'], 2, include_features)
        x.append(x_song)
        y.append(y_song)
    return np.vstack(x), np.hstack(y)


def train_test_model(x_train, x_test, y_train, y_test):
    """
    Creates keras sequential model, adding BiLSTM layers and trains it on x_train and x_test
    :param x_train: training data features
    :param x_test: testing data features
    :param y_train: target data for training
    :param y_test: target data for testing
    :return: loss and accuracy metrics
    """
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(units=64, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(units=32, return_sequences=False)))
    model.add(keras.layers.Dense(units=5, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=300)
    loss, accuracy = model.evaluate(x_test, y_test)

    return loss, accuracy


def main():
    """
    Main driver method that imports the music data and passes it to the ML method
    The commented out code adds data augmentation to the ML method or tries a different number of features
    :return: None
    """
    # with data augmentation:
    # df, songs = import_music(True)
    # train_songs, test_songs = train_test_split(songs, test_size=0.2, random_state=42)
    # augmented_data = create_fake_data(df, 10)
    # train_songs = train_songs + augmented_data
    # x_train, y_train = prepare_dataset(train_songs)
    # x_test, y_test = prepare_dataset(test_songs)
    # x_train = x_train.reshape(x_train.shape[0], 5, 1)
    # x_test = x_test.reshape(x_test.shape[0], 5, 1)
    # loss, accuracy = train_test_model(x_train, x_test, y_train, y_test)

    # without data augmentation
    # songs = import_music()
    # train_songs, test_songs = train_test_split(songs, test_size=0.2, random_state=42)
    # x_train, y_train = prepare_dataset(train_songs)
    # x_test, y_test = prepare_dataset(test_songs)
    # x_train = x_train.reshape(x_train.shape[0], 5, 1)
    # x_test = x_test.reshape(x_test.shape[0], 5, 1)
    # loss, accuracy = train_test_model(x_train, x_test, y_train, y_test)

    # without data augmentation, and all features
    songs = import_music()
    train_songs, test_songs = train_test_split(songs, test_size=0.2, random_state=42)
    x_train, y_train = prepare_dataset(train_songs, True)
    x_test, y_test = prepare_dataset(test_songs, True)
    x_train = x_train.reshape(x_train.shape[0], 5, 3)
    x_test = x_test.reshape(x_test.shape[0], 5, 3)
    loss, accuracy = train_test_model(x_train, x_test, y_train, y_test)

    print('Loss: {}, Accuracy: {}'.format(loss, accuracy))


if __name__ == "__main__":
    main()
