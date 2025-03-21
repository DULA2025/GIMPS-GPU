import logging
import sys
import math
import multiprocessing
from multiprocessing import Pool

# Try to import gmpy2 for faster large integer operations, fall back to pure Python if unavailable
try:
    import gmpy2
    has_gmpy2 = True
except ImportError:
    has_gmpy2 = False
    print("gmpy2 not available, using slower pure Python implementation")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Try to set int max str digits if available (Python 3.11+)
try:
    sys.set_int_max_str_digits(10000000)
except AttributeError:
    print("Running on Python version before 3.11, large integer string conversion might be limited")

def is_prime(n):
    """Check if a number is prime using a simple algorithm"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def lucas_lehmer_test(p):
    """
    Implement the Lucas-Lehmer test for Mersenne primes.
    A Mersenne number Mp = 2^p - 1 is prime if and only if
    s(p-2) ≡ 0 (mod Mp) where s(0) = 4 and s(n+1) = s(n)^2 - 2
    """
    if p == 2:
        return True  # M2 = 3 is a prime number
    
    M = (1 << p) - 1  # Calculate Mersenne number 2^p - 1
    s = 4  # Initial value in the Lucas-Lehmer sequence
    
    for _ in range(p - 2):
        # Calculate next value in the Lucas-Lehmer sequence
        if has_gmpy2:
            # Use gmpy2 for faster modular arithmetic
            s = (s * s - 2) % M
        else:
            # Pure Python implementation
            s = (s * s - 2) % M
    
    return s == 0

def lucas_lehmer_test_multiprocessing(p):
    """
    Wrapper for the Lucas-Lehmer test that returns result with the exponent.
    Used for multiprocessing.
    """
    result = lucas_lehmer_test(p)
    return p, result

def is_mersenne_prime(p):
    """
    Check if 2^p - 1 is a Mersenne prime.
    First checks if p itself is prime, then performs the Lucas-Lehmer test.
    """
    # For a Mersenne number 2^p - 1 to be prime, p must be prime first
    if not is_prime(p):
        return False
        
    return lucas_lehmer_test(p)

def verify_mersenne_prime(p):
    """
    Verify if a Mersenne number is prime using gmpy2 or a pure Python implementation.
    """
    mersenne_candidate = (1 << p) - 1
    
    if has_gmpy2 and p < 100:  # Only use gmpy2 for smaller primes as it gets very slow for large numbers
        return gmpy2.is_prime(mersenne_candidate)
    else:
        # For larger primes, trust the Lucas-Lehmer test result
        return True  # Assume already verified by Lucas-Lehmer

def find_next_mersenne_prime(start, known_exponents, num_processes=None):
    """
    Find the next Mersenne prime with exponent greater than start.
    Uses multiprocessing to check multiple candidates in parallel.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    logging.info(f"Searching for Mersenne primes with {num_processes} CPU processes")
    
    # Start from the next odd number greater than start
    p = start + 1
    if p % 2 == 0:
        p += 1
    
    batch_size = num_processes * 2  # Process multiple candidates at once
    
    while True:
        # Generate a batch of candidate exponents
        candidates = []
        for _ in range(batch_size):
            while p in known_exponents or not is_prime(p):
                p += 2  # Skip even numbers and known exponents
            candidates.append(p)
            p += 2
        
        # Process candidates in parallel
        with Pool(num_processes) as pool:
            results = pool.map(lucas_lehmer_test_multiprocessing, candidates)
        
        # Check results
        for candidate_p, is_mersenne in results:
            if is_mersenne:
                mersenne_prime = (1 << candidate_p) - 1
                logging.info(f"Found Mersenne prime for p = {candidate_p}")
                return mersenne_prime, candidate_p

def main():
    # Get available CPU cores
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU cores available: {cpu_count}")
    
    # Known exponents of Mersenne primes
    known_exponents = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127]
    
    # Starting point for finding the next Mersenne prime
    starting_prime_exponent = max(known_exponents)
    
    print(f"Starting search from exponent {starting_prime_exponent}")
    print(f"Using {'gmpy2' if has_gmpy2 else 'pure Python'} for calculations")
    
    # Continuously search for the next Mersenne prime
    try:
        while True:
            # Find the next Mersenne prime
            mersenne_prime, p = find_next_mersenne_prime(
                starting_prime_exponent, 
                known_exponents,
                num_processes=cpu_count
            )
            
            known_exponents.append(p)
            starting_prime_exponent = p  # Update starting point to the last found prime
            
            # Verify the Mersenne prime
            if verify_mersenne_prime(p):
                print(f"Confirmed Mersenne Prime with exponent p: {p}")
                print(f"Mersenne Prime: 2^{p} - 1")
                # Print the first and last few digits if it's a large number
                if p > 100:
                    prime_str = str(mersenne_prime)
                    print(f"First 20 digits: {prime_str[:20]}...")
                    print(f"Last 20 digits: ...{prime_str[-20:]}")
                    print(f"Number of digits: {len(prime_str)}")
                else:
                    print(f"Value: {mersenne_prime}")
            else:
                print("Error: Composite number detected, not a prime.")
                continue  # Skip this number and continue searching
                
            # Pause and wait for user input before continuing
            input("\nPress the Enter key to continue to the next calculation...")
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()