text = """{
  "name": "Darshil"
}"""
import ast
try:
    x = ast.literal_eval(text)
    print(x)
except Exception as e:
    print(e)
