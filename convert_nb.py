import json

with open('FORCAST_NEW.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        code_cells.append(source)

with open('notebook_logic.py', 'w', encoding='utf-8') as f:
    f.write('\n\n# CELL SEPARATOR\n\n'.join(code_cells))
