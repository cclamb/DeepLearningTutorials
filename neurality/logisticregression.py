__author__ = 'cclamb'

import numpy
import theano
import timeit

import theano.tensor as T


class LogisticRegression(object):
    def __init__(self, input, datapoints_dim, label_dim):
        # Here, we initialize an empty array with the dimentions
        # of the number of classes by the number of elements.
        #
        # In our MNIST example, this will yield a 768 by 10
        # two-dimensional shared numpy array.
        #
        # Named 'W' as it's the weights array. This is a 768x10
        # array as we will have a sequence of 768 possible connections
        # to each output node, and there's 10 output nodes. An
        # input value will have 768 elements.
        self.W = theano.shared(
            value=numpy.zeros(
                (datapoints_dim, label_dim),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        # These are bias values, and we'll use them later.
        # In our MNIST example, we will have one bias per class.
        self.b = theano.shared(
            value=numpy.zeros(
                (label_dim,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # An MNIST input value has 768 elements, so we can matrix
        # multiply with W (see T.dot(.) below). This yields a 10
        # element vector we can then add to the bias values (e.g. self.b).
        # Then, we use softmax to determine the distribution.
        #
        # As we take these values to be the probabilities that the submitted
        # input belongs to a given class, we call the resulting 10 element
        # vector the p_y_given_x (the probability that the input is a member
        # of class Y[i] given the probabilities in W biased by b).
        #
        # This is a symbolic function executed with Theano tensors.
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # This is the predicted class of the submitted input - essentially
        # the most likely class based on the probabilities obtained above.
        #
        # Note: this is a symbolic function, executed with Theano tensors.
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # This is essentially the model. This would usually be saved.
        self.params = [self.W, self.b]

    # Returns a symbolic function defining the NLL using Theano.
    def negative_log_likelihood(self, y):
        return -T.mean(
            T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        )

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def sgd_optimization_mnist(
        dataset,
        learning_rate=0.13,
        epoch_threshold=1000,
        batch_size=600,
        patience=5000,
        patience_increase=2,
        improvement_threshold=0.995):

    ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)) = dataset
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # Working with Logistic Regression
    #
    # Symbolic expression for the data
    index = T.lscalar()

    # These are symbolic variables that are frequently used in the below
    # function definitions.
    x = T.matrix('x')
    y = T.ivector('y')

    # This is a symbolically defined classifier...
    classifier = LogisticRegression(input=x, datapoints_dim=28*28, label_dim=10)

    # ...and the NLL as a cost function from that classifier.
    cost = classifier.negative_log_likelihood(y)

    # Gradients here are defined as dl/dW and dl/db here.
    gradient_W = T.grad(cost=cost, wrt=classifier.W)
    gradient_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [
        (classifier.W, classifier.W - learning_rate * gradient_W),
        (classifier.b, classifier.b - learning_rate * gradient_b)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_score = 0
    done_looping = False
    epoch = 0
    best_validation_loss = numpy.inf
    validation_frequency = min(n_train_batches, patience / 2)
    start_time = timeit.default_timer()
    iteration = 0
    while (epoch < epoch_threshold) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iteration = (epoch - 1) * n_train_batches + minibatch_index

            if (iteration + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set validation_losses = [validate_model(i)
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print 'epoch %i, minibatch %i/%i, validation error %f %%' % (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iteration * patience_increase)
                    best_validation_loss = this_validation_loss
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                print 'epoch %i, minibatch %i/%i, test error of best model %f %%' % (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    test_score * 100.
                )

        if patience <= iteration:
            done_looping = True
            break

    end_time = timeit.default_timer()
    print 'Optimization complete with best validation score of %f %% and test performance %f %%' % (
        best_validation_loss * 100.,
        test_score * 100.
    )

    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch,
        1. * epoch / (end_time - start_time)
    )
