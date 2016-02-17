iimport numpy
import time
import sys
import subprocess
import os
import random
import cPickle

from argparse import ArgumentParser
from is13.rnn.elman import model
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin,cue_contextwin

def load(fname):
    with open(os.path.join('/Users/ffancellu/git/is13/data/pickled_neg',fname),'rb') as f:
        train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-p',help="Pickled fname containing training,test and dev data")
    parser.add_argument('-f',help="Folder to store the log and best system")
    parser.add_argument('-c',help="Add cue related info",action='store_true')

    args = parser.parse_args()
    
    s = {'lr':0.014812362,
         'verbose':1,
         'decay':True, # decay on the learning rate if improvement stops
         'win':9, # number of words in the context window
         'bs':12, # number of backprop through time steps
         'nhidden':171, # number of hidden units
         'seed':345,
         'emb_dimension':81, # dimension of word embedding
         'nepochs':50,
         'folder':os.path.join('/Users/ffancellu/git/is13/log/elman',args.f)}

    folder = s['folder']
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    train_set, valid_set, test_set, dic = load(args.p)
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

    train_lex, train_y, train_cue = train_set
    valid_lex, valid_y, valid_cue = valid_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = vocsize,
                    de = s['emb_dimension'],
                    cs = s['win'],
                    cue = args.c)

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_lex,train_y,train_cue], s['seed'])
        print '[learning] epoch %d' % e
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            # take the context win of both
            # merge the results
            cwords = contextwin(train_lex[i], s['win'])
            words  = map(lambda x: numpy.asarray(x).astype('int32'),\
                         minibatch(cwords, s['bs']))
            if args.c:
                ccues = contextwin(train_cue[i],s['win'])
                cues_bs = map(lambda x: numpy.asarray(x).astype('int32'),\
                         minibatch(ccues, s['bs']))  
            labels = train_y[i]
            if not args.c:
                for word_batch , label_last_word in zip(words, labels):
                    rnn.train(word_batch, label_last_word, s['clr'])
                    rnn.normalize()
            else:
                for word_batch , cues_batch, label_last_word in zip(words, cues_bs,labels):
                    rnn.train(word_batch, cues_batch,label_last_word, s['clr'])
                    rnn.normalize()
                    rnn.normalize_cue()
            if s['verbose']:
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        if not args.c:
            predictions_test = [ map(lambda x: idx2label[x], \
                                 rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                                 for x in test_lex ]
            groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y]
            words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

            predictions_valid = [ map(lambda x: idx2label[x], \
                                 rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                                 for x in valid_lex ]
            groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y]
            words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]

            # evaluation // compute the accuracy using conlleval.pl
            res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
            res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

            if res_valid['f1'] > best_f1:
                rnn.save(folder)
                best_f1 = res_valid['f1']
                if s['verbose']:
                    print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20
                s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
                s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
                s['be'] = e
                subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
                subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
            else:
                print ''

        else:
            print "RESULTS ON VALIDATION SET...."   
            predictions_valid = [ map(lambda x: idx2label[x], \
                                 rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'),
                                    numpy.asarray(contextwin(c, s['win'])).astype('int32')))\
                                 for x,c in zip(valid_lex,valid_cue) ]
            groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y]
            words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]

            # evaluation // compute the accuracy using conlleval.pl  
            res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

            print "RESULTS ON TEST SET(S)..."
            for i,sub_set in enumerate(test_set):
                test_lex, test_y, test_cue = sub_set
                predictions_test = [ map(lambda x: idx2label[x], \
                                     rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'),
                                        numpy.asarray(contextwin(c, s['win'])).astype('int32')))\
                                     for x,c in zip(test_lex,test_cue) ]
                groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y]
                words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

                res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test%d.txt' % i)

            if res_valid['f1'] > best_f1:
                rnn.save(folder)
                best_f1 = res_valid['f1']
                if s['verbose']:
                    print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20
                s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
                s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
                s['be'] = e
                subprocess.call(['mv', folder + '/current.test0.txt', folder + '/best.test.txt'])
                subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
            else:
                print ''

        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5
        if s['clr'] < 1e-5: break

    print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder

