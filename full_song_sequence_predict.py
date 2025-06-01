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
keras module for doing data preprocessing - e.g. padding data
"""
from keras import preprocessing
"""
sklearn: machine learning package for python
Github: https://github.com/scikit-learn/scikit-learn
Main page: https://scikit-learn.org/stable/
GroupShuffleSplit is a module that separates training and testing data and also shuffles the data
"""
from sklearn.model_selection import GroupShuffleSplit
"""
pretty midi: Utility functions for handling MIDI data in a nice/intuitive way.
Authors: Colin Raffel and Daniel P. W. Ellis
Github: https://github.com/craffel/pretty-midi
Main page: https://craffel.github.io/pretty-midi/
"""
import pretty_midi


def import_music():
    """
    Reads in music xml files using music21 and extracts pitch, duration, offset, and fingering for each song and saves the data into a dataframe
    :return: dataframe of songs and features
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
                'pitch': element.pitch.nameWithOctave,
                'duration': element.duration.quarterLength,
                'offset': element.offset,
                'fingering': fingering,
            })

    # Create DataFrame
    df = pd.DataFrame(notes)
    return df


def create_fake_data(df, num_songs):
    """
    Creates fake songs by randomly selecting pitches and duration then calculates the fingering based on probability distribution and offset based on previous sum of durations
    resetting every 4 'beats' or time steps
    :param df: the original song data to calculate pitches and fingerings from
    :param num_songs: how many songs to generate
    :return: the original dataframe with the added songs
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
    fake_notes = []
    offset = 0.0
    # create new songs
    for i in range(num_songs):
        for j in range(0, 300):
            note = np.random.choice(unique_pitches)
            duration = np.random.choice(durations)
            if duration + offset > 4:
                duration = 4 - offset
            chosen_fingering = choose_fingering(distributions[note])
            fake_notes.append({
                'song': 'fake_song_' + str(i),
                'pitch': note,
                'duration': duration,
                'offset': offset,
                'fingering': chosen_fingering
            })
            offset += duration
            if offset >= 4:
                offset = 0.0

        df = pd.concat([df, pd.DataFrame(fake_notes)], ignore_index=True)
        fake_notes = []

    return df


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


def split_data(df):
    """
    Split data takes a data frame of song data and separates it into training and test sets.
    Also handles fake data, so it is only used during training
    :param df: dataframe of songs and features
    :return: training and test sets
    """
    # Group by song
    grouped = df.groupby('song')

    # Convert each song into a sequence
    x_sequences = [group[['pitch', 'duration', 'offset']].values for _, group in grouped]
    y_sequences = [group['fingering'].values for _, group in grouped]

    # Find max length for padding
    max_length = max(len(seq) for seq in x_sequences)

    # Pad sequences
    x_padded = preprocessing.sequence.pad_sequences(x_sequences, maxlen=max_length, dtype='float32', padding='post')
    y_padded = preprocessing.sequence.pad_sequences(y_sequences, maxlen=max_length, dtype='float32', padding='post')

    # Get unique song IDs
    song_ids = list(grouped.groups.keys())
    song_ids_without_augmentation = [id for id in song_ids if id.find('fake') == -1]

    # Split songs into train and test
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(song_ids_without_augmentation, groups=song_ids_without_augmentation))

    train_songs = [song_ids[i] for i in train_idx]
    test_songs = [song_ids[i] for i in test_idx]

    train_songs = np.append(train_songs, [id for id in song_ids if id.find('fake') != -1])

    # Select sequences based on song split
    x_train = np.array([x_padded[i] for i, song in enumerate(song_ids) if song in train_songs])
    y_train = np.array([y_padded[i] for i, song in enumerate(song_ids) if song in train_songs])
    x_test = np.array([x_padded[i] for i, song in enumerate(song_ids) if song in test_songs])
    y_test = np.array([y_padded[i] for i, song in enumerate(song_ids) if song in test_songs])

    return x_train, x_test, y_train, y_test


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
    model.add(layers.Bidirectional(layers.LSTM(units=32, return_sequences=True)))

    model.add(keras.layers.Dense(units=5, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=1000, batch_size=32)
    loss, accuracy = model.evaluate(x_test, y_test)

    return loss, accuracy


def main():
    """
    Main driver method that imports the music data and passes it to the ML method
    The commented out code adds data augmentation to the ML method
    :return: None
    """
    df = import_music()
    df['pitch'] = df['pitch'].apply(pitch_to_midi)

    # augmented_data = create_fake_data(df, 30)
    # x_train, x_test, y_train, y_test = split_data(augmented_data)
    # loss, accuracy = train_test_model(x_train, x_test, y_train, y_test)

    x_train, x_test, y_train, y_test = split_data(df)
    loss, accuracy = train_test_model(x_train, x_test, y_train, y_test)
    print('Loss: {}, Accuracy: {}'.format(loss, accuracy))


if __name__ == "__main__":
    main()
