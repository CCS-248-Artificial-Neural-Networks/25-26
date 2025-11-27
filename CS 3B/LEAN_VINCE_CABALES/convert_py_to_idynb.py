
import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

def py_to_ipynb(py_file):
	"""
	Converts a single .py file to an .ipynb file.
	The .ipynb file will have the same name as the .py file, with a single code cell containing the file's code.
	"""
	if not py_file.endswith('.py'):
		print("Input file must be a .py file.")
		return
	if not os.path.isfile(py_file):
		print(f"File not found: {py_file}")
		return
	with open(py_file, 'r', encoding='utf-8') as f:
		code = f.read()
	nb = new_notebook(cells=[new_code_cell(code)])
	ipynb_path = os.path.splitext(py_file)[0] + '.ipynb'
	with open(ipynb_path, 'w', encoding='utf-8') as f:
		nbformat.write(nb, f)
	print(f"Converted {os.path.basename(py_file)} to {os.path.basename(ipynb_path)}")


# Use the correct path to the Python file
py_file_path = os.path.join(os.path.dirname(__file__), "CABALES-EXERCISE_1.py")
py_to_ipynb(py_file_path)
