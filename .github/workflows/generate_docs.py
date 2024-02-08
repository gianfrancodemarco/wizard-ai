import ast
import os
import sys

from openai import OpenAI


def generate_docstring(text):
    # Function to generate docstring using OpenAI API
    # This function takes input `text` and returns the generated docstring
    try:
        client = OpenAI()

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You must generate a docstring for the following Python code:",
                },
                {
                    "role": "system",
                    "content": text
                }
            ],
            model="gpt-3.5-turbo",
        )

        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating docstring: {e}")
        return ""


def update_docstrings(filename):
    with open(filename, "r") as f:
        content = f.read()

    tree = ast.parse(content, filename=filename)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not ast.get_docstring(node):
                # Generate docstring for class or function
                generated_docstring = generate_docstring(content)
                if generated_docstring:
                    # Add the generated docstring to the node
                    ast.increment_lineno(node)
                    node.body.insert(0, ast.Expr(
                        value=ast.Constant(value=generated_docstring)))

    # Generate updated source code
    updated_code = ast.unparse(tree)

    # Write the updated content back to the file
    with open(filename, "w") as f:
        f.write(updated_code)


def update_docstrings_in_directory(scan_dir):
    # Scan directory for Python files
    for root, _, files in os.walk(scan_dir):
        for file in files:
            if file.endswith(".py"):
                filename = os.path.join(root, file)
                update_docstrings(filename)


if __name__ == "__main__":
    # Retrieve the SCAN_DIR argument from command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python generate_docs.py <SCAN_DIR>")
        sys.exit(1)
    SCAN_DIR = sys.argv[1]
    update_docstrings_in_directory(SCAN_DIR)
