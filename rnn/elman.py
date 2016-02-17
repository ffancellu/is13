import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class model(object):
    
    def __init__(self, nh, nc, ne, de, cs, cue = True):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                       (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        if cue:
            self.cue_emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,(2,de)).astype(theano.config.floatX))
            self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs * 2, nh)).astype(theano.config.floatX))
        else:
            self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # bundle
        if cue:
            self.params = [ self.emb, self.cue_emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
            self.names  = ['embeddings', 'cue_embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        else:
            self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
            self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']

        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        if cue: idxs_cues = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        if cue: c = self.cue_emb[idxs_cues].reshape((idxs_cues.shape[0],de*cs))
        y    = T.iscalar('y') # label

        if cue:

            def recurrence(x_t, c_t,h_tm1):
                h_t = T.nnet.sigmoid(T.dot(T.concatenate([x_t,c_t]), self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
                s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
                return [h_t, s_t]

            [h, s], _ = theano.scan(fn=recurrence, sequences=[x,c],
                outputs_info=[self.h0, None], n_steps=x.shape[0])

        else:

            def recurrence(x_t, h_tm1):
                h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
                s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
                return [h_t, s_t]

            [h, s], _ = theano.scan(fn=recurrence, \
                sequences=x, outputs_info=[self.h0, None], \
                n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.log(p_y_given_x_lastword)[y]
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))

        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        
        # theano functions
        # self.classify_keep_prob = theano.function(inputs=[idxs], outputs=p_y_given_x_sentence)
        if cue:
            self.classify = theano.function(inputs=[idxs,idxs_cues], outputs= y_pred)

            self.train = theano.function( inputs  = [idxs, idxs_cues, y, lr],
                                      outputs = nll,
                                      updates = updates )
        else:
            self.classify = theano.function(inputs=[idxs], outputs= y_pred)

            self.train = theano.function( inputs  = [idxs, y, lr],
                                          outputs = nll,
                                          updates = updates )

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

        self.normalize_cue = theano.function( inputs = [],
                         updates = {self.cue_emb:\
                         self.cue_emb/T.sqrt((self.cue_emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

