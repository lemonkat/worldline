import typing
import math
import string

# strings produced by the UID generator
UID = str

class UIDGenerator:
    def __init__(
        self, 
        state: int = 0,
        id_size: int = 4,
        id_chars: str = string.digits + string.ascii_uppercase,
        prime: int = 999983,
    ) -> None:
        """
        Initializes a Worldline UID Generator.
        Args:
            state (int, optional): The starting state for the generator. Defaults to 0.
            id_size: (int, optional): The length of the UIDs to produce. Defaults to 4.
            id_chars: (str, optional): The characters that will make up generated IDs. Defaults to all digits and all capital leters.
            prime: (int, optional): The prime number to use for UID scrambling. Defaults to 999983.
        Raises:
            ValueError: If `prime` and the maximum number of IDs (`len(id_chars) ** id_size`) are not coprime, as this would cause ID collisions.
        """
        self.id_size, self.id_chars = id_size, id_chars
        self.n_ids, self.prime = len(id_chars) ** id_size, prime
        
        if math.gcd(self.prime, self.n_ids) != 1:
            raise ValueError(f"CRITICAL ERROR: prime ({self.prime}) and n_ids ({self.n_ids}) are not coprime.")
            
        self.state = state

    def next(self, prefix: typing.Optional[str] = None) -> UID:
        """
        Gets the next available UID and increments the UID generator.
        Number theory go brrr.

        Args:
            prefix (str | None, optional): A prefix for the ID. Defaults to None (no prefix).

        Returns:
            str: a string of ID_SIZE characters from ID_CHARS, guaranteed to be unique for hopefully long enough. Format: [prefix]-[id] if a prefix is provided, else just [id].
        """
        self.state += 1
        scrambled = (self.state * self.prime) % self.n_ids
        id = "".join([self.id_chars[(scrambled // (len(self.id_chars) ** i)) % len(self.id_chars)] for i in range(self.id_size)])
        return id if prefix is None else f"{prefix}-{id}"
