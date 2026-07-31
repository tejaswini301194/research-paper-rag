import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import load_index
from src.qa import answer

store = load_index()
result = answer(store, "What is variance reduction used for in domain adaptation on streaming data?")

print("ANSWER:")
print(result["answer"])
print()
print("SOURCES:")
for s in result["sources"]:
    print(" -", s)
