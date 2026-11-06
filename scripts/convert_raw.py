with open('tools/build_master_notebook_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('nbf.v4.new_markdown_cell("""', 'nbf.v4.new_markdown_cell(r"""')

with open('tools/build_master_notebook_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated markdown cells to raw strings successfully!')
