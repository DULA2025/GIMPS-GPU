# Mersenne Prime Finder with CUDA

This Python script uses NVIDIA CUDA and the Lucas-Lehmer test to search for Mersenne primes (numbers of the form 2^p - 1 that are prime). It incorporates GPU acceleration for faster computation of large numbers.

## Features
- GPU-accelerated Lucas-Lehmer primality testing using CUDA
- Chunk-based processing for handling very large numbers
- Continuous search for new Mersenne primes
- Verification using the `gmpy2` library
- Logging of found Mersenne primes

## Requirements
- Python 3.x
- Numba (with CUDA support)
- NumPy
- gmpy2
- NVIDIA GPU with CUDA support

## Key Components

### Main Functions
1. `lucas_lehmer_kernel_chunked`: CUDA kernel performing the Lucas-Lehmer iteration
   - Processes numbers in chunks to handle large Mersenne numbers
   - Uses the formula: s = (s * s - 2) % M

2. `lucas_lehmer_test_cuda(p)`: Performs the Lucas-Lehmer test for exponent p
   - Returns True if 2^p - 1 is prime
   - Splits numbers into manageable chunks (30 bits each)

3. `find_next_mersenne_prime(start, known_exponents)`: 
   - Searches for the next Mersenne prime starting from given exponent
   - Uses modulo 6 optimization (primes are 1 or 5 mod 6)
   - Logs discovered primes

4. `verify_mersenne_prime(p)`: 
   - Additional verification using gmpy2's primality testing

### Helper Functions
- `split_chunks`: Splits large numbers into smaller chunks
- `combine_chunks`: Reconstructs numbers from chunks

## Usage
```python
# The script runs continuously, searching for new Mersenne primes
# Starting from the last known exponent
# Press Enter after each discovery to continue the search


Known Mersenne Prime Exponents
The script includes a list of known Mersenne prime exponents to skip:

    [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 2203, ...]
    Latest known: 136279841 (discovered by GIMPS/Luke Duran)

Implementation Details

    Uses 30-bit chunks to handle large numbers within int64 limits
    Employs CUDA with 32 threads per block
    Increases Python's integer string conversion limit
    Includes basic logging configuration

Running the Code

    Ensure all dependencies are installed
    Verify CUDA-capable GPU is available
    Run the script:

bash

python mersenne_finder.py

The program will search for new Mersenne primes, verify them, and wait for user input before continuing.
Notes

    Computation time increases significantly with larger exponents
    Memory usage grows with larger numbers
    Requires substantial GPU resources for large exponents
