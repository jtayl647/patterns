def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

def mod_inverse(e, phi):
    # Extended Euclidean Algorithm
    old_r, r = e, phi
    old_s, s = 1, 0

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s

    if old_s < 0:
        old_s += phi
    return old_s

def generate_keys(p, q):
    if not (is_prime(p) and is_prime(q)):
        raise ValueError("Both numbers must be prime.")
    elif p == q:
        raise ValueError("p and q cannot be the same.")

    n = p * q
    phi = (p - 1) * (q - 1)

    # Choose e
    e = 3
    while gcd(e, phi) != 1:
        e += 2  # try next odd number

    d = mod_inverse(e, phi)
    return (e, d, n)

def encrypt(m, e, n):
    return pow(m, e, n)

def decrypt(c, d, n):
    return pow(c, d, n)

if __name__ == "__main__":
    print("RSA Demo - Encrypt and Decrypt a Single Integer")
    
    # Choose primes
    p = int(input("Enter prime number p: "))
    q = int(input("Enter different prime number q: "))

    try:
        e, d, n = generate_keys(p, q)
        print(f"Public key (e, n): ({e}, {n})")
        print(f"Private key (d, n): ({d}, {n})")

        m = int(input(f"Enter message (as integer < {n}): "))
        if m >= n:
            raise ValueError("Message must be smaller than n.")

        c = encrypt(m, e, n)
        print(f"Encrypted: {c}")

        decrypted = decrypt(c, d, n)
        print(f"Decrypted: {decrypted}")
    except ValueError as ve:
        print("Error:", ve)