import tensorflow as tf

from common import EMBEDDING_SIZE, create_parser, get_data, process_vocabulary, predict
from perceptron import bag_of_words_perceptron
from mlp import bag_of_words_multilayer_perceptron
from rnn import rnn_model


def query():
    """Perform inference on some examples of documents from our classes."""
    tf.logging.set_verbosity(tf.logging.WARN)

    classes = get_data(FLAGS.data_dir, classes_only=True)
    FLAGS.output_dim = len(classes)

    QUERIES = [
        'University of Toronto',
        'TTC: Toronto Transit Commission, run buses, streetcars and subways',
        'Dragon',
        'Harley Davidson',
        'A kitten is a baby cat.',
        'Dog sleds are sleds pulled by a number of dogs on harnesses. They were used by the Eskimos.',
        'Bering Strait',
        'Whitehorse, Yukon',
        'Marijuana, also called hemp or cannabis',
        'Bat Out of Hell by Meat Loaf'
        ]
    _, x_query, _, _ = process_vocabulary(None, QUERIES, FLAGS, reuse=True)

    if FLAGS.model == 'perceptron':
        model = bag_of_words_perceptron
    elif FLAGS.model == 'mlp':
        model = bag_of_words_multilayer_perceptron
    elif FLAGS.model == 'rnn':
        model = rnn_model
    else:
        raise ValueError('unknown model')

    classifications = predict(x_query, model, FLAGS)
    for i, query in enumerate(QUERIES):
        print('The model classifies "{0}" as a member of the class {1}.'.format(
            query, classes['class'][classifications[i]]))


# Run script ##############################################
if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=EMBEDDING_SIZE,
        help='Number of dimensions in the embedding, '
             'i.e. the number of nodes in the hidden embedding layer (default: {})'.format(EMBEDDING_SIZE))
    parser.add_argument(
        'model_dir',
        help='The directory of the trained model')
    parser.add_argument(
        'model',
        help='Which model, e.g. perceptron, mlp, etc...')
    FLAGS = parser.parse_args()

    query()
