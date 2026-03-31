content = open('README.md', 'r', encoding='utf-8').read()

header = """---
title: SQL Debug Env
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

"""

if not content.startswith('---'):
    open('README.md', 'w', encoding='utf-8').write(header + content)
    print('Header added successfully!')
else:
    print('Header already exists')
    print('First 200 chars:', content[:200])