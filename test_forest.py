"""Quick test of the Forest multi-agent system with testpdf directory."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from agenticrag import Forest, GroqModel

# Create a local Forest (SQLite graph + local filesystem)
forest = Forest(
    verbose=True,
    model=GroqModel.LLAMA4_SCOUT,
)

# Add all PDFs from testpdf directory
print("=" * 60)
print("  Indexing all PDFs from testpdf/ ...")
print("=" * 60)

results = forest.add_directory(r"testpdf", pattern="*.pdf")

print(f"\n{'=' * 60}")
print(f"  Indexing Complete")
print(f"{'=' * 60}")
for r in results:
    status = "[OK]" if r.success else "[FAIL]"
    print(f"  {status} {r.file_name}")
    if r.success:
        print(f"       Title:  {r.title}")
        print(f"       Topics: {r.topics}")
        print(f"       Pages:  {r.page_count}")
    else:
        print(f"       Error:  {r.error}")

# Show forest info
print(f"\nForest: {forest.info()}")

# Interactive Q&A
print("\n" + "=" * 60)
print("  Forest Q&A  (type 'quit' to exit)")
print("=" * 60)

while True:
    try:
        question = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    if not question:
        continue

    print("Searching ...\n")
    answer = forest.ask(question)

    print(f"Answer: {answer.text}\n")
    print(f"  Confidence: {answer.confidence:.0%}")
    print(f"  Sources: {len(answer.sources)}")
    print(f"  Docs searched: {answer.documents_searched}")
    print(f"  Time: {answer.elapsed_seconds:.1f}s")
    if answer.was_rewritten:
        print(f"  [!] Answer was rewritten by Critic (hallucinations removed)")
