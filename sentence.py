class Sentence:

    def __init__(self, data, inference=False):
        self.weight = None
        self.features = get_features(data)

        if inference:
            self.goal = None
            self.phrase = data
        else:
            self.goal = data[:2]
            self.phrase = data[2:]


def get_features(data):
    words = set(data.split())

    return {

        "ends-with-en": ends_with("en", data), "ends-with-ed": ends_with("ed", data), "has-het": "het" in words,
        "has-een": "een" in words, "has-en": "en" in words, "has-de": "de" in words, "has-the": "the" in words,
        "has-be": "be" in words, "has-to": "to" in words, "has-of": "of" in words,
    }


def ends_with(letters, sentence):
    data = sentence.split()
    for word in data:
        if word.endswith(letters):
            return True
    return False


def parse_file(filenames, inference=False):
    tables = [[] for _ in filenames]

    for i in range(len(filenames)):
        filename = filenames[i]

        for line in open(filename, 'r', encoding='utf-8'):
            tables[i].append(Sentence(line,inference))

    return tables
