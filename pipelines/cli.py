import argparse
from .pipeline import DocumentPipeline

def main():
    parser = argparse.ArgumentParser(description="Ask questions about a PDF")
    parser.add_argument("pdf", help="path to the PDF document")
    args = parser.parse_args()
    pipeline = DocumentPipeline()
    pipeline.process_document(args.pdf)
    print("Document loaded. Ask questions; type 'exit' or 'quit' to stop.")
    try:
        while True:
            question = input("> ").strip()
            if question.lower() in ("quit", "exit"):
                break
            if question:
                print(pipeline.ask(question))
    except EOFError:
        return 0
    return 0

if __name__ == "__main__": main()
