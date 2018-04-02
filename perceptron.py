import tensorflow as tf

from common import WORDS_FEATURE, tic, toc, create_parser_training, parse_arguments, \
    preprocess_data, run_experiment, estimator_spec_for_softmax_classification

# Default values
MODEL_DIRECTORY = 'perceptron_model'
NUM_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.04


def bag_of_words_perceptron_model(features, labels, mode, params):
    """Perceptron architecture"""
    with tf.variable_scope('Perceptron'):
        bow_column = tf.feature_column.categorical_column_with_identity(
            WORDS_FEATURE, num_buckets=params.n_words)
        # Maps sequences of integers < params.n_words
        # to params.output_dim dimensional real-valued vectors
        # by taking the mean over the word (i.e. integer index) embedding values.
        bow_embedding_column = tf.feature_column.embedding_column(
            bow_column, dimension=params.output_dim)
        logits = tf.feature_column.input_layer(
            features,
            feature_columns=[bow_embedding_column])

    return estimator_spec_for_softmax_classification(logits, labels, mode, params)


def perceptron():
    """Train and evaluate the perceptron model."""
    tf.logging.set_verbosity(FLAGS.verbosity)

    print("Preprocessing data...")
    tic()
    train_raw, x_train, y_train, x_test, y_test, _, _, classes = preprocess_data(FLAGS)
    toc()

    # Set the output dimension according to the number of classes
    FLAGS.output_dim = len(classes)

    # Train and evaluate the model.
    tic()
    run_experiment(x_train, y_train, x_test, y_test,
                   bag_of_words_perceptron_model, 'train_and_evaluate', FLAGS)
    toc()


# Run script ##############################################
if __name__ == "__main__":
    parser = create_parser_training(MODEL_DIRECTORY, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)

    FLAGS = parse_arguments(parser)

    perceptron()
