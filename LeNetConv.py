import cPickle
import gzip
import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
                ),
            borrow=True
            )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
            )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
            )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W, self.b]

class HiddenLayer(object) :
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        if W is None :
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    ),
                dtype=theano.config.floatX
                )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
            )
        self.params = [self.W, self.b]


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out) :

        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
                ),
            name='W',
            borrow=True
            )

        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
                ),
            name='b',
            borrow=True
            )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
        self.params = [self.W,self.b]

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
                )
        if y.dtype.startswith('int') :
            return T.mean(T.neq(self.y_pred, y))
        else :
            raise NotImplementedError()


def load_data(dataset):
    dataset = r'mnist.pkl.gz'
    
#从"mnist.pkl.gz"里加载train_set, valid_set, test_set，它们都是包括label的
#主要用到python里的gzip.open()函数,以及 cPickle.load()。
#‘rb’表示以二进制可读的方式打开文件
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
   

#将数据设置成shared variables，主要时为了GPU加速，只有shared variables才能存到GPU memory中
#GPU里数据类型只能是float。而data_y是类别，所以最后又转换为int返回
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')


    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20,50], batch_size=500):
    rng  = numpy.random.RandomState(23455)

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    print '...building the model'

    layer0_input = x.reshape((batch_size, 1, 28, 28))
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2,2)
        )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2,2)
        )

    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] *4 *4,
        n_out=500,
        activation=T.tanh
        )

    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    cost = layer3.negative_log_likelihood(y)
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(  
        [index],  
        cost,  
        updates=updates,  
        givens={  
            x: train_set_x[index * batch_size: (index + 1) * batch_size],  
            y: train_set_y[index * batch_size: (index + 1) * batch_size]  
        }  
    )  

    print '...training'
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping) :
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches) :
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            if (iter + 1) % validation_frequency == 0:  
   
                validation_losses = [validate_model(i) for i  
                                     in xrange(n_valid_batches)]  
                this_validation_loss = numpy.mean(validation_losses)  
                print('epoch %i, minibatch %i/%i, validation error %f %%' %  
                      (epoch, minibatch_index + 1, n_train_batches,  
                       this_validation_loss * 100.))  
  
   
                if this_validation_loss < best_validation_loss:  
  
                      
                    if this_validation_loss < best_validation_loss *improvement_threshold:  
                        patience = max(patience, iter * patience_increase)  
  
                      
                    best_validation_loss = this_validation_loss  
                    best_iter = iter  
  
                     
                    test_losses = [  
                        test_model(i)  
                        for i in xrange(n_test_batches)  
                    ]  
                    test_score = numpy.mean(test_losses)  
                    print(('     epoch %i, minibatch %i/%i, test error of '  
                           'best model %f %%') %  
                          (epoch, minibatch_index + 1, n_train_batches,  
                           test_score * 100.))  
  
            if patience <= iter:  
                done_looping = True  
                break  
  
    end_time = time.clock()  
    print('Optimization complete.')  
    print('Best validation score of %f %% obtained at iteration %i, '  
          'with test performance %f %%' %  
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))  
    print >> sys.stderr, ('The code for file ' +  
                          os.path.split(__file__)[1] +  
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    
if __name__ == '__main__':
    evaluate_lenet5()
    
