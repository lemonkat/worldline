import pytest
from worldline.uid import UIDGenerator

def test_uid_generator_init():
    generator = UIDGenerator()
    assert generator.id_size == 4
    assert generator.prime == 999983
    assert generator.state == 0

def test_uid_generator_custom_init():
    generator = UIDGenerator(state=10, id_size=5, id_chars="01", prime=5)
    assert generator.id_size == 5
    assert generator.id_chars == "01"
    assert generator.prime == 5
    assert generator.state == 10
    assert generator.n_ids == 32

def test_uid_generator_coprime_error():
    # n_ids will be 2^2 = 4. Prime is 2. gcd(2, 4) == 2, which is not 1.
    with pytest.raises(ValueError, match="are not coprime"):
        UIDGenerator(id_size=2, id_chars="01", prime=2)

def test_next_uid_no_prefix():
    generator = UIDGenerator(id_size=4, id_chars="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", prime=999983)
    uid1 = generator.next()
    uid2 = generator.next()
    
    assert len(uid1) == 4
    assert len(uid2) == 4
    assert uid1 != uid2
    # Ensure they only contain valid characters
    for char in uid1 + uid2:
        assert char in generator.id_chars

def test_next_uid_with_prefix():
    generator = UIDGenerator()
    uid = generator.next(prefix="TEST")
    
    assert uid.startswith("TEST-")
    assert len(uid) == 9 # "TEST-" (5) + 4 random chars

def test_next_uid_uniqueness_and_determinism():
    # A small generator to test wrapping and uniqueness
    # n_ids = 3^2 = 9. prime 5 is coprime to 9.
    generator = UIDGenerator(id_size=2, id_chars="012", prime=5)
    
    generated_ids = set()
    for _ in range(9):
        generated_ids.add(generator.next())
        
    # All 9 IDs should be unique before it wraps around
    assert len(generated_ids) == 9
