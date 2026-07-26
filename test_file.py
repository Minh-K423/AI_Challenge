from clip_test import advanced_chunking

gen = advanced_chunking(list(range(100)))
print(gen)
print(type(gen))