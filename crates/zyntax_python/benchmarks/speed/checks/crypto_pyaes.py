import crypto_pyaes
import pyaes
import codecs
key = codecs.decode(b'a1f6258c877d5fcd8964484538bfc92c', 'hex')
iv = codecs.decode(b'ed62e16363638360fdd6ad62112794f0', 'hex')
aes = pyaes.new(key, pyaes.MODE_CBC, iv)
text = b"This is a test. What could possibly go wrong? " * 8
ciphertext = aes.encrypt(text)
print(len(ciphertext), ciphertext[:16].hex(), ciphertext[-16:].hex())
aes = pyaes.new(key, pyaes.MODE_CBC, iv)
print(aes.decrypt(ciphertext) == text)
ecb = pyaes.new(key, pyaes.MODE_ECB)
print(ecb.encrypt(b"0123456789abcdef").hex())
crypto_pyaes.benchmark()
print("ok")
