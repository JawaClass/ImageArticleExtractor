from textblob import TextBlob


def detect_language(string):
    b = TextBlob(string)
    return b.detect_language()
