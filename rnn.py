import tensorflow as tf

from common import EMBEDDING_DIM, WORD_METADATA_FILENAME, SENTENCE_METADATA_FILENAME, WORDS_FEATURE, \
    tic, toc, create_parser_training, parse_arguments, \
    preprocess_data, run_experiment, create_metadata, estimator_spec_for_softmax_classification

# Default values
MODEL_DIRECTORY = 'rnn_model'
NUM_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.002


def rnn_model(features, labels, mode, params):
    """RNN model to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], vocab_size=params.n_words, embed_dim=params.embed_dim)

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn.rnn_cell.GRUCell(params.embed_dim)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for softmax
    # classification over output classes.
    logits = tf.layers.dense(encoding, params.output_dim, activation=None)
    return estimator_spec_for_softmax_classification(logits, labels, mode, params)


def rnn():
    """Trains a multilayer perceptron with 1 hidden layer. It assumes that the data has already been preprocessed,
    e.g. by perceptron.py"""
    tf.logging.set_verbosity(FLAGS.verbosity)

    print("Preprocessing data...")
    tic()
    train_raw, x_train, y_train, x_test, y_test, classes = preprocess_data(FLAGS)
    toc()

    # Set the output dimension according to the number of classes
    FLAGS.output_dim = len(classes)

    # Train the RNN model.
    tic()
    run_experiment(x_train, y_train, x_test, y_test, rnn_model, 'train_and_evaluate', FLAGS)
    toc()

    # Create metadata for TensorBoard Projector.
    create_metadata(train_raw, classes, FLAGS)


# Run script ##############################################
if __name__ == "__main__":
    # Get common parser
    parser = create_parser_training(MODEL_DIRECTORY, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
    # Add command line parameters specific to this example
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=EMBEDDING_DIM,
        help='Number of dimensions in the embedding, '
             'i.e. the number of nodes in the hidden embedding layer (default: {})'.format(EMBEDDING_DIM))
    parser.add_argument(
        '--word-meta-file',
        default=WORD_METADATA_FILENAME,
        help='Word embedding metadata filename (default: {})'.format(WORD_METADATA_FILENAME))
    parser.add_argument(
        '--sent-meta-file',
        default=SENTENCE_METADATA_FILENAME,
        help='Sentence embedding metadata filename (default: {})'.format(SENTENCE_METADATA_FILENAME))

    FLAGS = parse_arguments(parser)

    rnn()
