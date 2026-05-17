"""
example_load_existing_tree.py

If you already have a saved PageIndex JSON, you can load it and start
asking questions immediately — no re-indexing needed.
"""

import json
from agenticrag import TreeSearcher, PageIndexConfig, extract_pages

# Load the saved tree
with open("my_document_tree.json") as f:
    tree = json.load(f)

# Reload the pages (needed for raw text retrieval)
pages = extract_pages("your_document.pdf")

config = PageIndexConfig(model="openai/gpt-oss-20b")
searcher = TreeSearcher(tree, config=config, pages=pages)

result = searcher.answer("What is the net revenue for 2023?")
print(result.text)
