import hashlib

def my_hash(s):
    # Return the MD5 hash of the input string as a hexadecimal string
    return hashlib.md5(s.encode()).hexdigest()


if __name__ == "__main__":
    my_hash('I love to hash it')

