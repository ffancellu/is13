import numpy
from sklearn import metrics

import codecs

# PREFIX = os.getenv('ATISDATA', '')

def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = codecs.open(filename,'w','utf8')
    f.writelines(out)
    f.close()
    
    return get_perf(filename)

def get_perf(filename):

    with codecs.open(filename,'r','utf8') as f:
        gs,ps = [],[]
        for line in f:
            line = line.strip()
            if line!='':
                w,g,p = line.split()
                if g=="I": gs.append(1)
                if g=="O": gs.append(0)
                
                if p=="I": ps.append(1)
                if p=="O": ps.append(0)             
    p,r,f1,s =  metrics.precision_recall_fscore_support(gs,ps)

    print metrics.classification_report(gs,ps)
    print metrics.confusion_matrix(gs, ps)
    # print {'p':res[0][0], 'r':res[1][0], 'f1':res[2][0]}

    return {'p':numpy.average(p,weights=s), 'r':numpy.average(r,weights=s), 'f1':numpy.average(f1,weights=s)}


if __name__ == '__main__':
    #print get_perf('valid.txt')
    print get_perf('/Users/ffancellu/git/is13/log/jordan/scope_neg_only_neg_elman_weights/best.test.txt')

