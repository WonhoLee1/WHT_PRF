import re

path = r'C:\SIMULIA\Documentation\2024LE\English\SIMACAEVERRefMap\simaver-c-elastic.htm'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

articles = text.split('abqversubsect')
sections_to_find = ['neo-Hookean', 'Yeoh', 'Arruda-Boyce']

for article in articles:
    match_title = re.search(r'<h2[^>]*>(.*?)</h2>', article)
    if not match_title:
        continue
    title = match_title.group(1)
    
    for s in sections_to_find:
        if s in title:
            print(f'Model: {s}')
            files = re.findall(r'<span class = \"ph ph abqinputfile\">(.*?)</span>', article)
            for f in files:
                print(f'  - {f}')
