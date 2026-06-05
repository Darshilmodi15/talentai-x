import ast

def process_file():
    with open('app/agents/parse_agent.py', 'r') as f:
        content = f.read()
    
    parsed = ast.parse(content)
    
    class Visitor(ast.NodeVisitor):
        def visit_Subscript(self, node):
            if isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    if "name" in node.slice.value:
                        print(f"Found ['{node.slice.value}'] at line {node.lineno}")
            self.generic_visit(node)
            
    Visitor().visit(parsed)

process_file()
