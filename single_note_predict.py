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
"""
collections: internal python module
docs: https://docs.python.org/3/library/collections.html
"""
from collections import defaultdict
"""
random: internal python module
docs: https://docs.python.org/3/library/random.html
"""
import random


def import_music():
    """
    Reads in music xml files using music21 and extracts pitch, duration, offset, and fingering for each song and saves the data into an array
    Notes are not separated by song
    :return: array of notes and their features
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
                'pitch': pitch_to_midi(element.pitch.nameWithOctave),
                'duration': float(element.duration.quarterLength),
                'offset': float(element.offset),
                'fingering': fingering,
            })

    return notes


def create_fake_data(notes, num_notes):
    """
    Creates fake notes by randomly selecting pitches then calculates the fingering based on probability distribution
    :param notes: the original notes data to calculate pitches and fingerings from
    :param num_notes: how many notes to generate
    :return: array of fake notes
    """
    # create dictionary to record fingering counts for all pitches
    pitch_fingering_counts = defaultdict(lambda: defaultdict(int))

    # count fingerings for each pitch
    for note in notes:
        pitch = note['pitch']
        fingering = note['fingering']
        pitch_fingering_counts[pitch][fingering] += 1

    # dictionary for fingering probabilities
    pitch_fingering_probs = {}

    # calculate fingering distributions
    for pitch, fingering_counts in pitch_fingering_counts.items():
        # Total occurrences of this pitch
        total = sum(fingering_counts.values())
        pitch_fingering_probs[pitch] = {fing: count / total for fing, count in fingering_counts.items()}

    # generate new notes based on above information
    fake_notes = []
    for i in range(num_notes):
        new_note = generate_new_pitch_fingering(pitch_fingering_probs)
        fake_notes.append(new_note)

    return fake_notes


def generate_new_pitch_fingering(pitch_fingering_probs):
    """
    Randomly choose pitch and generate new pitch fingering based on probability distribution
    :param pitch_fingering_probs: dictionary of pitches and fingering probabilities
    :return: new note and fingering
    """
    # choose pitch
    selected_pitch = random.choice(list(pitch_fingering_probs.keys()))
    # get fingerings and probabilities of pitch
    fingerings, probabilities = zip(*pitch_fingering_probs[selected_pitch].items())
    # choose fingering
    selected_fingering = random.choices(fingerings, weights=probabilities, k=1)[0]
    return {'pitch': selected_pitch, 'fingering': selected_fingering}


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


def prepare_dataset(notes):
    """
    Create list of notes and list of targets
    :param notes: notes dictionary
    :return: list of notes and targets
    """
    x, y = [], []
    for note in notes:
        x.append(note['pitch'])
        y.append(note['fingering'])
    return np.vstack(x), np.hstack(y)


def train_test_model(x_train, x_test, y_train, y_test):
    """
    Creates keras sequential model, adding dense layers and trains it on x_train and x_test
    :param x_train: training data features
    :param x_test: testing data features
    :param y_train: target data for training
    :param y_test: target data for testing
    :return: loss and accuracy metrics
    """
    model = keras.Sequential()
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(units=5, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=2000)
    loss, accuracy = model.evaluate(x_test, y_test)

    return loss, accuracy


def main():
    """
    Main driver method that imports the music data and passes it to the ML method
    The commented out code adds data augmentation to the ML method
    :return: None
    """
    # with data augmentation:
    # notes = import_music()
    # fake_notes = create_fake_data(notes, 500)
    # train_songs, test_songs = train_test_split(notes, test_size=0.2, random_state=42)
    # x_train, y_train = prepare_dataset(train_songs)
    # x_test, y_test = prepare_dataset(test_songs)
    #
    # x_fake, y_fake = prepare_dataset(fake_notes)
    # x_train = np.concatenate((x_train, x_fake), axis=0)
    # y_train = np.concatenate((y_train, y_fake), axis=0)
    #
    # loss, accuracy = train_test_model(x_train, x_test, y_train, y_test)

    # without data augmentation
    notes = import_music()
    train_songs, test_songs = train_test_split(notes, test_size=0.2, random_state=42)
    x_train, y_train = prepare_dataset(train_songs)
    x_test, y_test = prepare_dataset(test_songs)
    loss, accuracy = train_test_model(x_train, x_test, y_train, y_test)

    print('Loss: {}, Accuracy: {}'.format(loss, accuracy))


if __name__ == "__main__":
    main()
