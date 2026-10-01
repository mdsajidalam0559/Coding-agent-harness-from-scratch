import unittest

from textstats.stats import average_word_length, most_common_word, word_count


class Grade(unittest.TestCase):
    def test_word_count_handles_extra_spaces(self):
        self.assertEqual(word_count('hello   world  '), 2)

    def test_average_word_length(self):
        self.assertAlmostEqual(average_word_length('ab abcd'), 3.0)
        self.assertAlmostEqual(average_word_length('a ab'), 1.5)

    def test_average_word_length_empty(self):
        self.assertEqual(average_word_length(''), 0)

    def test_most_common_word(self):
        self.assertEqual(most_common_word('the cat and the hat'), 'the')
