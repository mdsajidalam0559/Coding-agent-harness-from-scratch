import argparse
import re
from collections import Counter


def top_words(text, n):
    words = re.findall(r"[a-z']+", text.lower())
    return Counter(words).most_common(n)


def main():
    parser = argparse.ArgumentParser(description="Show the most common words in a file.")
    parser.add_argument("file")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    with open(args.file) as f:
        pairs = top_words(f.read(), args.top)
    for word, count in pairs:
        print(f"{word}: {count}")


if __name__ == "__main__":
    main()
