import hashlib
import json


class Hash:
    def __init__(self):
        self.hash_value = ""
        return

    def hash_dict(self, dictionary: dict) -> str:
        """SHA256 hash of a dictionary."""
        dhash = hashlib.sha256()

        encoded = json.dumps(dictionary, sort_keys=True).encode()
        dhash.update(encoded)
        self.hash_value = dhash.hexdigest()
        return


if __name__ == "__main__":
    input_file = {"train": 1}
    # hash the input file
    a = Hash()
    a.hash_dict(input_file)
    print(a.hash_value)

    input_file = {"train": 0}
    # hash the input file
    a = Hash()
    a.hash_dict(input_file)
    print(a.hash_value)
