
class Any2Int:

    def __init__(self, min_count: int, include_UNK: bool, include_PAD: bool):
        self.min_count = min_count
        self.include_UNK = include_UNK
        self.include_PAD = include_PAD
        self.frozen = False
        self.UNK_i = -1
        self.UNK_s = "<UNK>"
        self.PAD_i = -2
        self.PAD_s = "<PAD>"
        self.voc_size = 0
        self._s2i = dict()
        self._i2s = []
        self.frequency = dict()

    def iter_item(self):
        return enumerate(self._i2s)

    def get_s2i(self, s, default: int):
        assert self.frozen
        i = self._s2i.get(s, -1)
        if i >= 0:
            return i
        elif self.include_UNK:
            return self.UNK_i
        else:
            return default

    def __getitem__(self, s):
        return self.s2i(s)

    def s2i(self, s):
        i = self.get_s2i(s, -1)
        if i >= 0:
            return i
        else:
            raise Exception(f"out of vocabulary entry {s}")

    def contains(self, s):
        return self.get_s2i(s, -1) != -1

    def i2s(self, i):
        assert self.frozen
        if 0 <= i < self.voc_size:
            return self._i2s[i]
        else:
            raise Exception(f"not entry at position {i} for a vocabulary of size {self.voc_size}")

    def add_to_counts(self, s):
        assert not self.frozen
        self.frequency[s] = self.frequency.get(s, 0)+1

    def freeze(self):
        assert not self.frozen
        if self.include_UNK:
            self.UNK_i = len(self._i2s)
            self._i2s.append(self.UNK_s)
        if self.include_PAD:
            self.PAD_i = len(self._i2s)
            self._i2s.append(self.PAD_s)
        for s, count in sorted(self.frequency.items(), key=lambda x: -x[1]):
            if count >= self.min_count:
                self._i2s.append(s)
        for i, s in enumerate(self._i2s):
            self._s2i[s] = i
        self.voc_size = len(self._i2s)
        self.frozen = True

    def __reduce__(self):
        return Any2Int, (2, self.include_UNK, self.include_PAD), (self.min_count, self.include_UNK, self.frozen,
                                                                  self.UNK_i, self.UNK_s, self.PAD_i, self.PAD_s,
                                                                  self.voc_size, self._s2i, self._i2s, self.frequency)

    def __setstate__(self, state):
        self.min_count = state[0]
        self.include_UNK = state[1]
        self.frozen = state[2]
        self.UNK_i = state[3]
        self.UNK_s = state[4]
        self.PAD_i = state[5]
        self.PAD_s = state[6]
        self.voc_size = state[7]
        self._s2i = state[8]
        self._i2s = state[9]
        self.frequency = state[10]

