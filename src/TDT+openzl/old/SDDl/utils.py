import numpy as np

import struct
def generate_smooth_array(n, type=np.float32):
    import numpy as np
    a = np.linspace(0, 1, n).astype(type)
    return a


# generate a 2D array with a given shape of smooth values
def generate_smooth_2d_array(shape, lower_bound=0, upper_bound=1, type=np.float32):
    import numpy as np
    a = np.linspace(lower_bound, upper_bound, shape[0] * shape[1]).astype(type)
    return a.reshape(shape)


# generate 2D array of smooth oscillating values
def generate_oscillating_2d_array(shape, lower_bound=0, upper_bound=1, type=np.float32):
    a = np.linspace(lower_bound, upper_bound, shape[0] * shape[1]).astype(type)
    a = np.sin(a)
    return a.reshape(shape)



# store a 2D array in a tsv file
def store_tsv_file(file_name, array):
    #header = "signal id\t"
    #header += "\t".join([str(i) for i in range(array.shape[1])])
    # create an array of signal ids
    signal_ids = np.arange(array.shape[0]).reshape(-1, 1)
    # concatenate signal ids and the array
    array = np.concatenate((signal_ids, array), axis=1)
    np.savetxt(file_name, array, delimiter="\t")



# generate a 2D numpy array of boolean values and its equivalent float32 array
def generate_boolean_array(n_rows, n_cols):
    bool_array = np.random.choice([True, False], size=(n_rows, n_cols))
    float_array = bool_array_to_float32(bool_array)
    return bool_array, float_array


# a function to take a 2D bool array and retuns its decimal representation
def bool_to_int(bool_array):
    """
    packbit pads the last byte with zeros if the number of bits is not a multiple of 8
    :param bool_array:
    :return:
    """
    byte_array = np.packbits(bool_array.flatten())
    return int.from_bytes(byte_array, byteorder='big')

def char_to_bool(char_array):
    bool_array = np.array([1 if x == b'1' else 0 for x in char_array.flatten()], dtype='bool')
    return bool_array.reshape(char_array.shape)


def bool_array_to_float32(bool_array):
    ba_flatten = bool_array.flatten()
    # a float array with size of len(ba_flatten)//32
    float_array = np.zeros(len(ba_flatten)//32, dtype='float32')
    for i in range(0, len(ba_flatten), 32):
        float_array[i//32] = bool_to_int(ba_flatten[i:i+32])
    return float_array

# a function to take a decimal number and return its 2D bool array representation
def int_to_bool(int_val, m, n):
    byte_array = int_val.to_bytes((m*n)//8, byteorder='big')
    bool_array = np.unpackbits(np.frombuffer(byte_array, dtype='uint8'))
    return bool_array.reshape(m, n)


def bool_to_binary_string(bool_array):
    return ''.join(['1' if x else '0' for x in bool_array.flatten()])

def bool_array_to_float321(bit_array, n):
    # Ensure bit_array is a boolean array
    bit_array = np.asarray(bit_array, dtype=np.bool_)
    # Pack bits into bytes
    byte_array = np.packbits(bit_array)
    # Convert byte array to float32
    # Calculate the required padding to make the byte array length a multiple of 4
    bytes_needed = ((len(bit_array) + 7) // 8)
    if bytes_needed % 4 != 0:
        padding = 4 - (bytes_needed % 4)
        byte_array = np.pad(byte_array, (0, padding), 'constant')
    float_array = np.frombuffer(byte_array, dtype=np.float32)
    return float_array

def int_to_bool1(value, m, n):
    # Convert the integer to a binary string, zero-padded to the required length
    bit_string = bin(value)[2:].zfill(m * n)
    # Convert the binary string to a boolean array
    bit_array = np.array([int(bit) for bit in bit_string], dtype=np.bool_)
    return bit_array

def bool_to_int1(bit_array):
    # Convert boolean array to binary string
    bit_string = ''.join(str(int(bit)) for bit in bit_array)
    # Convert binary string to integer
    value = int(bit_string, 2)
    return value

def float32_to_bool_array1(float_array, m, n):
    # Convert float32 array back to byte array
    byte_array = float_array.tobytes()
    # Convert byte array to bit array
    bit_array = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8))
    # Return only the relevant bits (m * n)
    return bit_array[:m * n]
#######################
def floats_to_bool_arrays(float_list):
    def float_to_bool_array(f):
        # Convert a float to its binary representation
        binary = format(np.float32(f).view(np.int32), '032b')
        # Convert the binary string to a list of boolean values
        return [bit == '1' for bit in binary]

    # Convert each float in the list to a boolean array
    bool_arrays = [float_to_bool_array(f) for f in float_list]
    # Return as a numpy array
    return np.array(bool_arrays, dtype=bool)


def tuple_to_string(t):
    res = '( '
    for ele in t:
        res += '('
        for i in ele:
            res += str(i)
            res += ','
        res += '), '
    res += ')'
    return res

def list_to_string(l):
    res = '('
    for ele in l:
        res += str(ele)
        res += ','
    res += ')'
    return res


def compute_entropy(data_set):
    from collections import Counter
    import numpy as np
    # Count the occurrences of each byte value
    counts = Counter(data_set)
    # Calculate the probability of each byte value
    probs = np.array(list(counts.values())) / len(data_set)
    # Calculate the entropy using the formula
    entropy = -np.sum(probs * np.log2(probs))
    return entropy



def find_max_consecutive_similar_values(data_set):
    # Find the maximum number of consecutive similar values
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(data_set)):
        if data_set[i] == data_set[i - 1]:
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
        else:
            current_consecutive = 1
    return max_consecutive
# Inside modeling/utils.py
def generate_partitions(elements):
    if len(elements) == 1:
        yield [elements]
        return
    first = elements[0]
    for smaller in generate_partitions(elements[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller
