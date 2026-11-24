def add_bad(item, bucket=[]):
    bucket.append(item)
    return bucket

def add_good(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket

add_bad("a")
print(add_bad("b"))

add_good("a")
print(add_good("b"))