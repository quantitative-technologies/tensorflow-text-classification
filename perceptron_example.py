import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from common import get_data, extract_data, process_vocabulary, run_experiment, tic, toc, \
    create_parser_training, parse_arguments
from perceptron import bag_of_words_perceptron

# Default values
MODEL_DIRECTORY = 'perceptron_example_model'
NUM_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.005


def perceptron_example():
    """Perceptron example demonstrating online learning, and also evaluation separate from training."""
    tf.logging.set_verbosity(FLAGS.verbosity)

    train_raw, test_raw, classes = get_data(FLAGS.data_dir)

    # Set the output dimension according to the number of classes
    FLAGS.output_dim = len(classes)

    print("\nSplitting the training and test data into two pieces...")
    # Seeding necessary for reproducibility.
    np.random.seed(FLAGS.np_seed)

    # Shuffle data to make the distribution of classes roughly stratified after splitting.
    train_raw = shuffle(train_raw)
    test_raw = shuffle(test_raw)

    train1_raw, train2_raw = np.split(train_raw, 2)
    test1_raw, test2_raw = np.split(test_raw, 2)

    print("First split:")
    x_train1_sentences, y_train1, x_test1_sentences, y_test1 = extract_data(train1_raw, test1_raw)

    print("\nProcessing the vocabulary...")
    tic()
    x_train1, x_test1, vocab_processor, n_words = process_vocabulary(x_train1_sentences, x_test1_sentences, FLAGS)
    toc()

    # Train the model on the first split.
    tic()
    run_experiment(x_train1, y_train1, x_test1, y_test1, bag_of_words_perceptron, 'train_and_evaluate', FLAGS)
    toc()

    # Next we perform incremental training with the 2nd half of the split data.
    print("\nSecond split extends the vocabulary.")
    x_train2_sentences, y_train2, x_test2_sentences, y_test2 = extract_data(train2_raw, test2_raw)

    # Extend vocab_processor with the newly added training vocabulary, and save the vocabulary processor for later use.
    tic()
    x_train2, x_test2, vocab_processor, n_words = process_vocabulary(x_train2_sentences, x_test2_sentences, FLAGS,
                                                                     reuse=False, vocabulary_processor=vocab_processor,
                                                                     extend=True)
    toc()

    # Train the model on the second split.
    tic()
    run_experiment(x_train2, y_train2, x_test2, y_test2, bag_of_words_perceptron, 'train_and_evaluate', FLAGS)
    toc()

    # We may be interested in the model performance on the training data (e.g. to evaluate removable bias).
    print("\nEvaluation of the model performance on the training data.:")
    run_experiment(None, None, x_train1, y_train1, bag_of_words_perceptron, 'evaluate', FLAGS)


# Run script ##############################################
if __name__ == "__main__":
    # Get common parser
    parser = create_parser_training(MODEL_DIRECTORY, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
    FLAGS = parse_arguments(parser)

    perceptron_example()
