import tensorflow as tf

from common import EMBEDDING_DIM, WORD_METADATA_FILENAME, WORDS_FEATURE, \
    tic, toc, create_parser_training, parse_arguments, \
    preprocess_data, run_experiment, create_metadata, estimator_spec_for_softmax_classification

# Default values
MODEL_DIRECTORY = 'rnn_model'
NUM_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.002


def rnn_model(features, labels, mode, params):
    """RNN model architecture using GRUs."""
    with tf.variable_scope('RNN'):
        # This creates an embedding matrix of dimension (params.n_words, params.embed_dim).
        # Thus an integer input matrix of size (num_docs, max_doc_len) will get mapped
        # to a tensor of dimension (num_docs, max_doc_len, params.embed_dim).
        word_vectors = tf.contrib.layers.embed_sequence(
            features[WORDS_FEATURE], vocab_size=params.n_words, embed_dim=params.embed_dim)

        # Unpack word_vectors into a sequence of length max_doc_len,
        # of tensors of dimension (num_docs, params.embed_dim),
        # so that the RNN is given the n-th word of the document at the n-th step.
        word_sequence = tf.unstack(word_vectors, axis=1)

        # Create a Gated Recurrent Unit cell with hidden layer size params.embed_dim.
        cell = tf.nn.rnn_cell.GRUCell(params.embed_dim)

        # Create an unrolled Recurrent Neural Networks of length params.max_doc_len,
        # providing the length of each sequence (i.e. number of words in each document)
        # so that the output from the last element get propagated to the output layer.
        _, encoding = tf.nn.static_rnn(cell, word_sequence, dtype=tf.float32,
                                       sequence_length=features['LENGTHS_FEATURE'])

        # The output layer
        logits = tf.layers.dense(encoding, params.output_dim, activation=None)

    return estimator_spec_for_softmax_classification(logits, labels, mode, params)


def rnn():
    """Trains a multilayer perceptron with 1 hidden layer. It assumes that the data has already been preprocessed,
    e.g. by perceptron.py"""
    tf.logging.set_verbosity(FLAGS.verbosity)

    print("Preprocessing data...")
    tic()
    train_raw, x_train, y_train, x_test, y_test, train_lengths, test_lengths, classes \
        = preprocess_data(FLAGS, sequence_lengths=True)
    toc()

    # Set the output dimension according to the number of classes
    FLAGS.output_dim = len(classes)

    # Train the RNN model.
    tic()
    run_experiment(x_train, y_train, x_test, y_test, rnn_model,
                   'train_and_evaluate', FLAGS, train_lengths, test_lengths)
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
 
    FLAGS = parse_arguments(parser)

    rnn()
