try:
    some_dict = {'a': 1}
    some_dict.pop('\n  \"name\"')
except Exception as e:
    print(type(e).__name__, repr(str(e)))
