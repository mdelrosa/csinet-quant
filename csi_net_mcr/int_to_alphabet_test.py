import sys
import torch
import unicodedata
import numpy as np


sys.path.append("/home/mdelrosa/git/ArithmeticEncodingPython")
import pyae

from decimal import getcontext
getcontext().prec = 45

def make_alphabet(L):
    """ contruct alphabet of length L, remove whitespace and control characters """
    true_i = 0
    alphabet = ''
    while(len(alphabet) < L):
        while(str.isspace(chr(true_i)) or unicodedata.category(chr(true_i))[0] == "C"):
            true_i += 1
        alphabet += chr(true_i)
        true_i += 1
    return alphabet

def return_symbol(x, alphabet):
    """ return x-th symbol from alphabet """
    return alphabet[x]

vect_return_symbol = np.vectorize(return_symbol)

n_batch = 2
n_features = int(512 / 4)
# max_msg_size = 4
max_msg_size = int(n_features / 16)
L = 1024
# L = 1024

alphabet = make_alphabet(L)

# define random counts dict
counts = {}
for symbol in alphabet:
    counts[symbol] = np.random.randint(20)+1

q = torch.randint(L, size=(n_batch, n_features))
q_symbols = q.detach().numpy()
q_symbols = vect_return_symbol(q_symbols, alphabet)
q_messages = []
for array in q_symbols:
    q_messages.append("".join(array))

AE = pyae.ArithmeticEncoding(frequency_table=counts)
                    # save_stages=True)

print(f"--- n_features: {n_features} - max_msg_size: {max_msg_size} ---")
bit_counts = []
for i, original_msg_full in enumerate(q_messages):
    preamble_str = f"-> #{i}:"
    # Encode the message
    print(f"Full Message to Encode: {original_msg_full}")
    idx = 0
    decoded_msg_full = ""
    bits_full = 0
    while idx < len(original_msg_full):
        idx_e = min([idx+max_msg_size, n_features])
        original_msg = original_msg_full[idx:idx_e] 
        encoded_msg, encoder, interval_min_value, interval_max_value = AE.encode(msg=original_msg, 
                                                                                probability_table=AE.probability_table)
        print(f"{preamble_str} Partial Encoded Message: {encoded_msg}")

        # Get the binary code out of the floating-point value
        binary_code, encoder_binary = AE.encode_binary(float_interval_min=interval_min_value,
                                                    float_interval_max=interval_max_value)
        print(f"{preamble_str} Partial binary code: {binary_code} with len={len(binary_code)}")

        # Decode the message
        decoded_msg, decoder = AE.decode(encoded_msg=encoded_msg, 
                                        msg_length=len(original_msg),
                                        probability_table=AE.probability_table)
        decoded_msg = "".join(decoded_msg)
        print(f"{preamble_str} Partial Decoded Message: {decoded_msg}")
        decoded_msg_full += decoded_msg
        idx += max_msg_size
        bits_full += len(binary_code)

    bit_counts.append(bits_full)
    print(f"{preamble_str} Message Decoded Successfully? {original_msg_full == decoded_msg_full}")

print(bit_counts)