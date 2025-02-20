import numba
import numpy as np
import numba.cuda as cuda
import logging
import sys
import gmpy2

# Increase the limit for integer string conversion
sys.set_int_max_str_digits(10000000)  # Increase this limit as necessary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Kernel to perform the Lucas-Lehmer iteration for smaller chunks
@cuda.jit
def lucas_lehmer_kernel_chunked(s_array, p, M_chunks, chunk_size):
    idx = cuda.grid(1)
    if idx == 0:
        s = 4
        M = 0
        for i in range(len(M_chunks)):
            M += M_chunks[i] << (i * chunk_size)
        for _ in range(p - 2):
            s = (s * s - 2) % M
        s_array[0] = s

def split_chunks(number, chunk_size):
    chunks = []
    while number:
        chunks.append(number & ((1 << chunk_size) - 1))
        number >>= chunk_size
    return chunks

def combine_chunks(chunks, chunk_size):
    number = 0
    for chunk in reversed(chunks):
        number = (number << chunk_size) | chunk
    return number

def lucas_lehmer_test_cuda(p):
    if p == 2:
        return True
    
    s_array = cuda.device_array(1, dtype=np.int64)
    chunk_size = 30  # Define chunk size (must be small enough to fit in int64)
    M = (1 << p) - 1
    M_chunks = split_chunks(M, chunk_size)
    
    M_chunks_device = cuda.to_device(np.array(M_chunks, dtype=np.int64))
    
    threads_per_block = 32
    blocks_per_grid = (len(M_chunks) + threads_per_block - 1) // threads_per_block
    
    lucas_lehmer_kernel_chunked[blocks_per_grid, threads_per_block](s_array, p, M_chunks_device, chunk_size)
    
    s_result = s_array.copy_to_host()
    
    return s_result[0] == 0

def find_next_mersenne_prime(start, known_exponents):
    p = start
    while True:
        p += 1
        if p in known_exponents:
            continue
        
        # Modulo 6 check for primes congruent to 1 or 5 modulo 6
        if p % 6 not in {1, 5}:
            continue
        
        # Perform the Lucas-Lehmer test for the current exponent
        if lucas_lehmer_test_cuda(p):
            mersenne_prime = (1 << p) - 1
            logging.info(f"Found Mersenne prime for p = {p}")
            return mersenne_prime, p

# Known exponents of Mersenne primes (add the latest Mersenne prime to continue the search 136279841 found by: GIMPS / Luke Duran - PRP / Gpuowl on NVIDIA A100)
known_exponents = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 2203,  
]

# Starting point for finding the next Mersenne prime
starting_prime_exponent = max(known_exponents)

def verify_mersenne_prime(p):
    mersenne_candidate = (1 << p) - 1
    return gmpy2.is_prime(mersenne_candidate)

# Continuously search for the next Mersenne prime
while True:
    mersenne_prime, p = find_next_mersenne_prime(starting_prime_exponent, known_exponents)
    known_exponents.append(p)
    starting_prime_exponent = p  # Update starting point to the last found prime

    # Verify the Mersenne prime using gmpy2
    if verify_mersenne_prime(p):
        print(f"Confirmed Mersenne Prime with exponent p: {p}")
    else:
        print("Error: Composite number detected, not a prime.")
        continue  # Skip this number and continue searching

    # Pause and wait for user input before continuing
    input("\nPress the Enter key to continue to the next calculation...")
