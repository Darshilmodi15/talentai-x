import ast
import os

def check_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        parsed = ast.parse(content)
        
        class Visitor(ast.NodeVisitor):
            def visit_Subscript(self, node):
                if isinstance(node.slice, ast.Constant):
                    if isinstance(node.slice.value, str):
                        if "name" in node.slice.value:
                            print(f"{filepath}:{node.lineno}: found ['{node.slice.value}']")
                self.generic_visit(node)
                
        Visitor().visit(parsed)
    except Exception as e:
        pass

for root, _, files in os.walk('app/'):
    for file in files:
        if file.endswith('.py'):
            check_file(os.path.join(root, file))
