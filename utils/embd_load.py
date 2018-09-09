# coding=utf-8
import numpy as np
from configs.hyper_params import TrainConfig


def list2array(_list):
    return np.array(list(map(float, _list)))


def array2string(array):
    return ' '.join(map(str, array.tolist()))


class EmbdMapper:
    def __init__(self, config):
        self.data_path = config.embd_path

        # special char
        self.unk_char = config.unk_char

        # load embedding
        self.num, self.dim, self.embd_dict = self._load_embd()

    def _load_embd(self):
        embd_dict = dict()
        with open(self.data_path, "r") as f:
            # read meta data
            meta_line = f.readline()
            _, dim = meta_line.split()
            dim = int(dim)
            while 1:
                lines = f.readlines(1000)
                if not lines:
                    break
                for line in lines:
                    chars, *scalars = line[:-1].split()
                    if len(chars) > 1:
                        continue
                    assert (len(scalars) == dim)
                    array = list2array(scalars)
                    embd_dict[chars] = array
        # add special chars
        zeros = np.zeros(shape=(dim,), dtype=np.float32)
        embd_dict[self.unk_char] = zeros
        num = len(embd_dict.keys())
        print('number of vector:{}, dimension:{}'.format(num, dim))
        return num, dim, embd_dict

    def char2embd(self, char):
        if char not in self.embd_dict.keys():
            raise KeyError
        return self.embd_dict[char]

    def text2embd(self, text):
        return [self.char2embd(char) for char in text]

    def get_char2idx(self):
        char2idx = {char: idx for idx, char in enumerate(self.embd_dict.keys())}
        return char2idx

    def get_vocab(self):
        return list(self.embd_dict.keys())

    def get_lookup_table(self):
        # return [self.embd_dict[k] for k in self.embd_dict.keys()]
        # return list(self.embd_dict.values())
        return np.array(list(self.embd_dict.values()))

    def save(self, path):
        embd_list = list()
        for k in self.embd_dict.keys():
            embd_list.append('{} {}\n'.format(k, array2string(self.embd_dict[k])))
        sorted(embd_list)
        with open(path, 'w', newline='\n') as f:
            f.writelines(['{} {}\n'.format(self.num, self.dim)])
            f.writelines(embd_list)


def main():
    config = TrainConfig()
    em = EmbdMapper(config)
    lookup_table = em.get_lookup_table()
    print(lookup_table.shape)
    # em.save('data/embd/sgns.renmin.char.reduce')


if __name__ == '__main__':
    main()
