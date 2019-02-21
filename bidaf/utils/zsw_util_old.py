import numpy as np


def pad_answer(batch):
    output = []
    length_info = [len(x[0]) for x in batch]
    max_length = max(length_info)
    for one in batch:
        output.append([x + [0] * (max_length - len(x)) for x in one])
    return output



def padding(sequence, pads=0, max_len=None, dtype='int32', return_matrix_for_size=False):
    # we should judge the rank
    if True or isinstance(sequence[0], list):
        v_length = [len(x) for x in sequence]  # every sequence length
#        seq_max_len = max(v_length)
#        if (max_len is None) or (max_len > seq_max_len):
#            max_len = seq_max_len
        v_length = list(map(lambda z: z if z <= max_len else max_len, v_length))
        x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
        xMask = (np.ones((len(sequence), max_len)) * False).astype(bool)
        for idx, s in enumerate(sequence):
            trunc = s[:max_len]
            x[idx, :len(trunc)] = trunc
            xMask[idx,:len(trunc)] = np.asarray([False]*len(trunc))
        if return_matrix_for_size:
            v_matrix = np.asanyarray([map(lambda item: 1 if item < line else 0, range(max_len)) for line in v_length],
                                     dtype=dtype)
            return x, v_matrix
        return x, np.asarray(v_length, dtype='int32'),xMask
    else:
        seq_len = len(sequence)
        if max_len is None:
            max_len = seq_len
        v_vector = sequence + [0] * (max_len - seq_len)
        padded_vector = np.asarray(v_vector, dtype=dtype)
        v_index = [1] * seq_len + [0] * (max_len - seq_len)
        padded_index = np.asanyarray(v_index, dtype=dtype)
        return padded_vector, padded_index

def padding_char(sequence,pads=0, max_len=None):
      char_size = len(sequence[0][0])
      x = (np.ones((len(sequence), max_len,char_size)) * pads).astype('int32')
      for idx,s in enumerate(sequence):
          trunc = s[:max_len]
          x[idx,:len(trunc)] = trunc
      return x
def shuffle_data(data, axis='pWordId'):
    pool = {}
    for one in data:
        length = len(one[axis])  ## doc len
        if length not in pool:
            pool[length] = []
        pool[length].append(one)
    for one in pool:
        np.random.shuffle(pool[one])
    length_lst = list(pool.keys())
    np.random.shuffle(length_lst)
    return [x for y in length_lst for x in pool[y]]
