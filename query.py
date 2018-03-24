import numpy as np
import tensorflow as tf

from common import EMBEDDING_DIM, create_parser, parse_arguments, get_data, process_vocabulary, predict
from perceptron import bag_of_words_perceptron_model
from mlp import bag_of_words_MLP_model
from rnn import rnn_model

# Default values
QUERY_FILENAME = 'queries.txt'


def query():
    """Perform inference on some examples of documents from our classes."""
    tf.logging.set_verbosity(FLAGS.verbosity)

    classes = get_data(FLAGS.data_dir, classes_only=True)
    FLAGS.output_dim = len(classes)

    queries = np.loadtxt(FLAGS.query_file, dtype=str, delimiter='\n')
    _, x_query, _, _ = process_vocabulary(None, queries, FLAGS, reuse=True)

    if FLAGS.model == 'perceptron':
        model = bag_of_words_perceptron_model
    elif FLAGS.model == 'mlp':
        model = bag_of_words_MLP_model
    elif FLAGS.model == 'rnn':
        model = rnn_model
    else:
        raise ValueError('unknown model')

    classifications = predict(x_query, model, FLAGS)
    for i, query in enumerate(queries):
        print('The model classifies "{}" as a member of the class {}.'.format(
            query, classes['class'][classifications[i]]))


# Run script ##############################################
if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=EMBEDDING_DIM,
        help='Number of dimensions in the embedding, '
             'i.e. the number of nodes in the hidden embedding layer (default: {})'.format(EMBEDDING_DIM))
    parser.add_argument(
        'model_dir',
        help='The directory of the trained model')
    parser.add_argument(
        'model',
        help='Which model, e.g. perceptron, mlp, etc...')
    parser.add_argument(
        '--query-file',
        default=QUERY_FILENAME,
        help='Name of the queries file (default: {})'.format(QUERY_FILENAME))
    parser.add_argument(
        '--checkpoint',
        type=int,
        default=None,
        help='Model checkpoint to query. By default the most recent model is used.')
    FLAGS = parse_arguments(parser)

    query()
