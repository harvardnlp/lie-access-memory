'''Example:
    ``import data.generate as gen``
    When run at the top level,
    ``gen.generate_all('copy', 'data/copy', 32,
                        1000, 2, 64,
                        100, 65, 128,
                        37)
    ``
    generates a dataset for the copy task at `data/copy`,
    with batch size 32;
    with 1000 training batches of min size 2, max size 64;
    with 100 validation batches of min size 65, max size 128;
    with alphabet size 37.
    The files generated include human readable form text filess (.txt)
    and an hdf5 file for easy loading into torch/python, along with
    translation dicts corresponding strings to numbers.

    Similarly,
    ``gen.generate_all('double', 'data/double', 32,
                        1000, 2, 64,
                        100, 65, 128)
    ``
    generates a dataset for the double task at `data/double`,
    with batch size 32,
    with 1000 training batches of min size 2, max size 64;
    with 100 validation batches of min size 65, max size 128;
    with implicit alphabet size 10, encoding 0 to 9.

    For repeat tasks,
    ``gen.generate_all('repeatCopy', 'data/prioritySort', 32,
                        1000, 20, 20,
                        100, 20, 20,
                        37,
                        train_nrepeat_low=2, train_nrepeat_high=10,
                        valid_nrepeat_low=11, valid_nrepeat_high=20)
    ``
    generates a dataset for repeatCopy,
    with batch size 32;
    with 1000 training batches of size 20, for repeating 2 to 10 times;
    with 100 validation batches of size 20, for repeating 11 to 20 times;
    with alphabet size 37.
'''
import numpy as np
import h5py, os
import random
import json
from collections import OrderedDict
from itertools import izip
START = '^'
SEP = '#'
END = '$'


PERM = ['copy', 'reverse', 'bigramFlip', 'oddFirst']
ARITH = ['addition', 'mult', 'double']
OTHER = ['repeatCopy', 'prioritySort']

def fmt(n, seq_len):
    return ('{:0' + str(seq_len) + 'd}').format(n)
def mapstr(m, seq_len, reverse_=False):
    v = reverse_ and -1 or 1
    return [fmt(s, seq_len)[::v] for s in m]
def interleave(a, b):
    return "".join(i for j in izip(a, b) for i in j)
def getaddition(batch_size, seq_len):
    '''`seq_len` indicates the length of each summand.'''
    r = [random.randint(0, 10**seq_len - 1) for _ in xrange(batch_size)]
    s = [random.randint(0, 10**seq_len - 1) for _ in xrange(batch_size)]
    t = [a + b for (a, b) in izip(r, s)]
    return ([interleave(fmt(_r, seq_len)[::-1], fmt(_s, seq_len)[::-1]) for
                (_r, _s) in izip(r, s)],
            mapstr(t, seq_len + 1, reverse_=True))
def getmult(batch_size, seq_len, interleave_=False):
    r = [random.randint(0, 10**seq_len - 1) for _ in xrange(batch_size)]
    s = [random.randint(0, 10**seq_len - 1) for _ in xrange(batch_size)]
    t = [a * b for (a, b) in izip(r, s)]
    if interleave_:
        return ([interleave(fmt(_r, seq_len), fmt(_s, seq_len)) for
                    (_r, _s) in izip(r, s)],
                mapstr(t, seq_len * 2))
    else:
        return ([fmt(_r, seq_len) + '*' + fmt(_s, seq_len) for
                    (_r, _s) in izip(r, s)],
                mapstr(t, seq_len * 2))
def getdouble(batch_size, seq_len):
    '''seq_len` indicates the length of the input number.'''
    r = [random.randint(0, 10**seq_len - 1) for _ in xrange(batch_size)]
    t = [2 * a for a in r]
    return mapstr(r, seq_len), mapstr(t, seq_len+1)

def seq2str(seq, sep=','):
    return sep.join(map(str, seq))
def getprioritySort(batch_size, seq_len, low=0, high=10, tostr=False):
    out = np.random.randint(low, high, (batch_size, seq_len))
    inp = [np.random.permutation(list(enumerate(l))).reshape(-1) for l in out]
    if tostr:
        return map(seq2str, inp), map(seq2str, out)
    else:
        return inp, out
def getinput(batch_size, seq_len, low=0, high=10, tostr=False, prepend_seqlen=False):
    '''samples a tensor of integers from `low` to `high`,
    with size (`batch_size`, `seq_len`).
    Return this tensor if not `tostr` else return a list of strings,
    with str_i being the comma-separated version of each row of tensor.'''
    tensor = np.random.randint(low, high, (batch_size, seq_len))
    if prepend_seqlen:
        tensor = np.append([[seq_len]]*batch_size, tensor, axis=1)
    if not tostr:
        return tensor
    else:
        return map(seq2str, tensor)

def getcopy(batched_in):
    return batched_in

def getrepeatCopy(batched_in, n):
    return np.tile(batched_in, n)

def getreverse(batch_in):
    return [s[::-1] for s in batch_in]

def getoddFirst(batch_in):
    return [list(s[::2]) + list(s[1::2]) for s in batch_in]

def _ngramflip(seq, n):
    return sum([list(seq[i+n-1:(i-1 if i > 0 else None):-1])
            for i in range(0, len(seq), n)],
               [])

def getbigramFlip(batch_in):
    '''`batch_in` should be a tensor fo size (batch_size, seq_len)'''
    return [_ngramflip(row, n=2) for row in batch_in]

def getbatch(batch_size, seq_len, task, start=START, end=END, sep=SEP, alphabet=10,
            nrepeat=None):
    gettask = globals()['get' + task]
    def _format(batch_in, batch_out):
        return ','.join([
            ','.join([start] + [seq2str(in_)] + [sep] + [seq2str(out)] + [end])
            for in_, out in izip(batch_in, batch_out)])
    if task in PERM:
        batch_in = getinput(batch_size, seq_len, high=alphabet)
        batch_out = gettask(batch_in)
        return _format(batch_in, batch_out)
    elif task == 'prioritySort':
        batch_in, batch_out = gettask(batch_size, seq_len, high=alphabet)
        samples = []
        for b in range(batch_size):
            sample = []
            for i in range(seq_len):
                pri = batch_in[b][2*i]
                val = batch_in[b][2*i + 1]
                sample += [start] * pri + [str(val)]
            samples.append(','.join([start] + sample + [sep, seq2str(batch_out[b]), end]))
        return ','.join(samples)
    elif task == 'repeatCopy':
        batch_in = getinput(batch_size, seq_len, high=alphabet)
        assert nrepeat is not None, 'nrepeat should not be None'
        batch_out = getrepeatCopy(batch_in, nrepeat)
        return ','.join([
            ','.join([start] * (nrepeat+1) + [seq2str(in_)] + [sep] + [seq2str(out)] + [end])
            for in_, out in izip(batch_in, batch_out)])
    elif task in ARITH:
        batch_in, batch_out = gettask(batch_size, seq_len)
        return ''.join([start + sep.join([in_, out]) + end
            for in_, out in izip(batch_in, batch_out)])
    else:
        raise Exception('unknown task')

def generate_set(batch_size, nbatch, min_len, max_len, task, alphabet=10, tofile=None,
                nrepeat_low=None, nrepeat_high=None):
    '''`min_len` and `max_len` are inclusive bounds on the input size.
    `nrepeat_low` and `nrepeat_high` are inclusive bounds on repeatCopy repeat size.'''
    batches = []
    for _ in xrange(nbatch):
        seq_len = np.random.randint(min_len, max_len+1)
        if task in ('bigramFlip', 'oddFirst'):
            # truncate seq_len to largest even number
            seq_len = seq_len // 2 * 2
        nrepeat = None
        if nrepeat_low and nrepeat_high:
            nrepeat = np.random.randint(nrepeat_low, nrepeat_high + 1)
        batches.append(getbatch(batch_size, seq_len, task, alphabet=alphabet,
                                nrepeat=nrepeat))
    dataset = '\n'.join(batches)
    if task in PERM + ['repeatCopy', 'prioritySort']:
        assert not nrepeat_high or nrepeat_high <= alphabet, \
                'repeatCopy repeat size should be at most the alphabet size'
        translate = OrderedDict([
                (str(i), i) for i in range(alphabet)
            ])
        translate[START] = alphabet + 0
        translate[SEP] = alphabet + 1
        translate[END] = alphabet + 2
    elif task in ARITH:
        translate = OrderedDict([
                (str(i), i) for i in range(10)
            ])
        translate[START] = 10
        translate[SEP] = 11
        translate[END] = 12
        translate['+'] = 13
        translate['*'] = 14
    else:
        raise ValueException('unknown task')
    if tofile:
        with open(tofile, 'w') as f:
            f.write(dataset)
        translate2file(translate, tofile + '.tsl')
    return dataset, translate

def str2tensor(str_dataset, tr, oneindex=True, sep=SEP, end=END):
    line1, _, _ = str_dataset.partition('\n')
    batch_size = line1.count(sep)
    total_samples = str_dataset.count(sep)
    nbatches = total_samples // batch_size
    csv = ',' in str_dataset
    if not csv:
        symbols_per_row, remainder = divmod(
            len(str_dataset) - str_dataset.count('\n'),
            batch_size)
    else:
        symbols_per_row, remainder = divmod(
            str_dataset.count(',') + str_dataset.count('\n') + 1,
            batch_size)
    assert remainder == 0, 'remainder={} is nonzero'.format(remainder)
    # symbols_per_row+1 because of the final end marker
    tensor = np.zeros([batch_size, symbols_per_row])
    c = 0
    if oneindex:
        tr = {key: val + 1 for key, val in tr.iteritems()}
    for line in str_dataset.split('\n'):
        chars = csv and line.split(',') or list(line)
        samples = []
        i = 0
        while chars:
            if chars[i] == end:
                samples.append(chars[:i+1])
                chars = chars[i+1:]
                i = 0
            else:
                i = i+1
        # print(samples)
        for j in xrange(len(samples[0])):
            for b in xrange(batch_size):
                tensor[b, c] = tr[samples[b][j]]
            c += 1
    return tensor

def _tensor2str(tensor, translate, csv=False, oneindex=True):
    '''takes a tensor that represents one batch (with END markers) and
    convert it to str.'''
    m = csv and ',' or ''
    return m.join([
            m.join([translate[k] for k in tensor[b]])
        for b in range(tensor.shape[0])])
def tensor2str(tensor, translate, csv=False, oneindex=True, end=END):
    tr = {val + int(oneindex): key
        for key, val in translate.iteritems()}
    s = ''
    lastc = 0
    for c in range(tensor.shape[1]):
        if tensor[0, c] == translate[end] + int(oneindex):
            s += _tensor2str(tensor[:, lastc:c+1], tr, csv=csv, oneindex=oneindex)
            if c < tensor.shape[1] - 1:
                s += '\n'
            lastc = c + 1
    return s

def translate2file(d, file):
    '''assuming values are indices.'''
    with open(file, 'w') as f:
        for k, v in d.iteritems():
            f.write('{}\t{}\n'.format(k, v))

def generate_all(
        task, dest, batch_size,
        train_nbatch, train_min_len, train_max_len,
        valid_nbatch, valid_min_len, valid_max_len,
        alphabet=10,
        train_nrepeat_low=None, train_nrepeat_high=None,
        valid_nrepeat_low=None, valid_nrepeat_high=None,
        oneindex=True):
    STR2INT = 'str2int.txt'
    TENSOR = 'tensor.hdf5'
    METADATA = 'metadata.json'
    meta = OrderedDict()
    meta['task'] = task
    meta['batch_size'] = batch_size
    meta['alphabet_size'] = alphabet
    meta['min_train_seq_len'] = train_min_len
    meta['max_train_seq_len'] = train_max_len
    meta['min_valid_seq_len'] = valid_min_len
    meta['max_valid_seq_len'] = valid_max_len
    meta['train_dataset_nbatches'] = train_nbatch
    meta['valid_dataset_nbatches'] = valid_nbatch
    if train_nrepeat_high:
        meta['train_nrepeat_high'] = train_nrepeat_high
    if train_nrepeat_low:
        meta['train_nrepeat_low'] = train_nrepeat_low
    if valid_nrepeat_high:
        meta['valid_nrepeat_high'] = valid_nrepeat_high
    if valid_nrepeat_low:
        meta['valid_nrepeat_low'] = valid_nrepeat_low
    def mapjoin(s, L):
        return s.join(map(str, L))
    if task in ARITH:
        alphabet = 10
    trainname = mapjoin('_', [train_nbatch, train_min_len, train_max_len])
    validname = mapjoin('_', [valid_nbatch, valid_min_len, valid_max_len])
    if train_nrepeat_low:
        trainname += '%{}%{}'.format(train_nrepeat_low, train_nrepeat_high)
    if valid_nrepeat_low:
        validname += '%{}%{}'.format(valid_nrepeat_low, valid_nrepeat_high)
    foldername = mapjoin('-', [
        task, batch_size, alphabet, trainname, validname])
    import os
    try:
        os.mkdir(dest)
    except OSError as e:
        # print e.strerror
        # continue
        pass
    try:
        os.mkdir(os.path.join(dest, foldername))
    except OSError as e:
        # print e.strerror
        # continue
        pass
    def pjoin(file):
        return os.path.join(dest, foldername, file)
    with open(pjoin(METADATA), 'w') as f:
        json.dump(meta, f, indent=4)
    trainstr, tr = generate_set(batch_size, train_nbatch,
        train_min_len, train_max_len, task, alphabet=alphabet,
        nrepeat_low=train_nrepeat_low, nrepeat_high=train_nrepeat_high,
        tofile=pjoin(mapjoin('_', [task, batch_size, train_nbatch,
                        train_min_len, train_max_len]) + '.train.txt'))
    validstr, _ = generate_set(batch_size, valid_nbatch,
        valid_min_len, valid_max_len, task, alphabet=alphabet,
        nrepeat_low=valid_nrepeat_low, nrepeat_high=valid_nrepeat_high,
        tofile=pjoin(mapjoin('_', [task, batch_size, valid_nbatch,
                        valid_min_len, valid_max_len]) + '.valid.txt'))
    str2int_txt = pjoin(STR2INT)
    tr_saved = {k: v+int(oneindex) for k, v in tr.iteritems()}
    translate2file(tr_saved, str2int_txt)
    traintensor = str2tensor(trainstr, tr, oneindex=oneindex)
    validtensor = str2tensor(validstr, tr, oneindex=oneindex)
    with h5py.File(pjoin(TENSOR), 'w') as f:
        trainset = f.create_dataset('train', traintensor.shape, dtype='i')
        validset = f.create_dataset('valid', validtensor.shape, dtype='i')
        trainset[...] = traintensor
        validset[...] = validtensor
    return os.path.join(dest, foldername)
def _reconstruct_tensor(trainstr, validstr, traintsl, validtsl, outfile=None,
            oneindex=True):
    traintensor = str2tensor(trainstr, traintsl, oneindex=oneindex)
    validtensor = str2tensor(validstr, validtsl, oneindex=oneindex)
    if outfile:
        with h5py.File(outfile, 'w') as f:
            trainset = f.create_dataset('train', traintensor.shape, dtype='i')
            validset = f.create_dataset('valid', validtensor.shape, dtype='i')
            trainset[...] = traintensor
            validset[...] = validtensor
            print 'saving tensor to', outfile
    return traintensor, validtensor

def reconstruct_tensor(datadir, outfile=None, oneindex=True):
    fnames = {}
    for f in os.listdir(datadir):
        key = ''
        if 'train' in f:
            key += 'train'
        elif 'valid' in f:
            key += 'valid'
        if key:
            if f.endswith('.txt'):
                key += 'txt'
            elif f.endswith('.tsl'):
                key += 'tsl'
            fnames[key] = f
    import os.path as P
    strs = {}
    for k, v in fnames.iteritems():
        with open(P.join(datadir, v), 'r') as fl:
            strs[k] = fl.read()
    for k in ['traintsl', 'validtsl']:
        strs[k] = dict([s.split('\t') for s in strs[k].split('\n') if s != ''])
        strs[k] = {k: int(v) for k, v in strs[k].iteritems()}
    if outfile is True:
        outfile = P.join(datadir, 'tensor.hdf5')
    return _reconstruct_tensor(strs['traintxt'], strs['validtxt'],
                               strs['traintsl'], strs['validtsl'],
                               outfile=outfile, oneindex=oneindex)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
        'generate datasets and reconstruct hdf5 files')
    parser.add_argument('--makehdf5', action='store',
        help='reconstruct hdf5 from the text files in the given data dir')
    parser.add_argument('--makeallhdf5', action='store_true',
        help='reconstruct all tensor.hdf5 files in each data dir in data/')
    parser.add_argument('task', type=str, nargs='?')
    parser.add_argument('dest', type=str, nargs='?')
    parser.add_argument('num_args', type=int, nargs='*')
    args = parser.parse_args()
    datadir = args.makehdf5
    if args.makeallhdf5:
        print 'reconstructing all tensors'
        for root, dirs, files in os.walk('data/'):
            for dir_ in dirs:
                dir_ = os.path.join(root, dir_)
                print dir_
                try:
                    reconstruct_tensor(dir_, outfile=True)
                except Exception as e:
                    print 'not a data dir'
                else:
                    print 'tensor reconstructed'
    elif datadir:
        print 'reconstructing tensor'
        reconstruct_tensor(datadir, outfile=True)
    else:
        print 'constructing dataset'
        generate_all(args.task, args.dest, *args.num_args)
