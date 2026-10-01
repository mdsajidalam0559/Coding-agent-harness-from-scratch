def word_count(text):
    return len(text.split(' '))


def average_word_length(text):
    words = text.split()
    return sum(len(w) for w in words) // len(words)


def most_common_word(text):
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return min(counts, key=counts.get)
