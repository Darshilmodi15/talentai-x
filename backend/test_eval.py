import ast
try:
    ast.literal_eval("{\n  \"name\": \"John Doe\"\n}")
except Exception as e:
    print(type(e).__name__, repr(str(e)))
