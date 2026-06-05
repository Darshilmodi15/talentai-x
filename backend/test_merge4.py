try:
    some_dict = {}
    print(some_dict["\n  \"name\""])
except Exception as e:
    print(type(e).__name__, repr(str(e)))
