__author__ = 'cclamb'

import gzip
import theano
import numpy
import cPickle

import theano.tensor as T

class InputPreprocessor(object):
    def __init__(self, filename):
        self.filename = filename

    # Opening the zipped file and extracting the contents.
    def load_data(self):
        if self.filename is None:
            raise NameError('Filename not set')
        # Opening the zipped file and extracting the contents.
        with gzip.open(self.filename, 'rb') as f:
            # Loading the pickled data from the compressed file.
            # The file is formated as a 3-tuple of tuples,
            # where each tuple consists of a list of 10000
            # data elements (the normalized images) and
            # 10000 class labels (a number, 0-9, that
            # indicates the class of the associated data element).
            #
            # Each data element is a normalized 28x28 MNIST
            # image rendered in a single list of 784 elements
            # (e.g. 28 x 28).
            #
            # So, to extract the image from the train_set tuple,
            # you'd use something like:
            #
            # images = train_set[0]
            # classes = train_set[1]
            # first_image = images[0]
            # first_image_class = classes[0]
            #
            # Loading the tuples from the 3-tuple file format,
            # and closing the archive. This could be done with
            # the 'with' statement as well.
            self.train_set, self.valid_set, self.test_set = cPickle.load(f)


    # This function will extract the images from the
    # images classes, and place that information into
    # two separate shared theano variables.
    #
    # We use shared variables because When storing data
    # on the GPU it has to be stored as floats
    # therefore we will store the labels as 'floatX'
    # as well ('shared_y' does exactly that). But
    # during our computations we need them as ints
    # (we use labels as index, and if they are
    # floats it doesn't make sense) therefore
    # instead of returning 'shared_y' we will have
    # to cast it to int. This little hack lets us
    # get around this issue.
    #
    # You can tune the size of the mini-batches to fit
    # in available GPU memory.
    @staticmethod
    def _shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(
            numpy.asarray(
                data_x,
                dtype=theano.config.floatX
            )
        )
        shared_y = theano.shared(
            numpy.asarray(
                data_y,
                dtype=theano.config.floatX
            )
        )
        return shared_x, T.cast(shared_y, 'int32')

    def process_data(self):
        # Create the extraced data.
        test_set_x, test_set_y = self._shared_dataset(self.test_set)
        valid_set_x, valid_set_y = self._shared_dataset(self.valid_set)
        train_set_x, train_set_y = self._shared_dataset(self.train_set)
        return [
            (train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)
        ]
