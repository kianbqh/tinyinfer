"""CPU-only contiguous KV reference. No numeric library dependency."""
from copy import deepcopy


def exact_int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low},{high}]')
    return value


class KVCache:
    """Storage [slot][layer][position]; each value represents [head,dim].

    append_batch is transactional across all slots/layers. Caller supplies one
    position sequence per layer for each slot. Deep copies establish ownership.
    """
    def __init__(self, slots, layers, capacity):
        self.slots = exact_int(slots, 'slots', 1, 1024)
        self.layers = exact_int(layers, 'layers', 1, 1024)
        self.capacity = exact_int(capacity, 'capacity', 1, 1024)
        self.lengths = [0] * slots
        self.k = [[[] for _ in range(layers)] for _ in range(slots)]
        self.v = [[[] for _ in range(layers)] for _ in range(slots)]

    def _slot(self, slot):
        return exact_int(slot, 'slot', 0, self.slots - 1)

    def reset(self, slot):
        slot = self._slot(slot)
        self.lengths[slot] = 0
        self.k[slot] = [[] for _ in range(self.layers)]
        self.v[slot] = [[] for _ in range(self.layers)]

    def append_batch(self, slots, keys, values):
        if not slots or len(slots) != len(keys) or len(slots) != len(values):
            raise ValueError('nonempty matched batch required')
        checked = [self._slot(s) for s in slots]
        if len(set(checked)) != len(checked):
            raise ValueError('duplicate slot')
        counts = []
        for s, k, v in zip(checked, keys, values):
            if len(k) != self.layers or len(v) != self.layers:
                raise ValueError('one key/value sequence per layer required')
            count = len(k[0])
            if count <= 0 or any(len(x) != count for x in list(k) + list(v)):
                raise ValueError('matched positive append lengths required')
            if self.lengths[s] + count > self.capacity:
                raise ValueError('cache capacity exceeded')
            counts.append(count)
        staged = deepcopy((keys, values))
        for s, k, v, count in zip(checked, staged[0], staged[1], counts):
            for layer in range(self.layers):
                self.k[s][layer].extend(k[layer])
                self.v[s][layer].extend(v[layer])
            self.lengths[s] += count

    def read(self, slot, layer):
        slot = self._slot(slot)
        layer = exact_int(layer, 'layer', 0, self.layers - 1)
        return deepcopy((self.k[slot][layer], self.v[slot][layer]))


def causal_keys(query_index, query_offset, valid_length):
    exact_int(query_index, 'query_index', 0, 1023)
    exact_int(query_offset, 'query_offset', 0, 1023)
    exact_int(valid_length, 'valid_length', 0, 1024)
    return range(min(valid_length, query_offset + query_index + 1))
