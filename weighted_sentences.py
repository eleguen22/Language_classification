class WeightedSentences:

    def __init__(self, sentences):
        self.sentences = sentences
        self.total = 0

        for sentence in sentences:
            sentence.weight = 1 / len(sentences)
            self.total += sentence.weight

    def change_weight(self, index, weight):

        self.total = self.total - self.sentences[index].weight
        self.sentences[index].weight = weight
        self.total = self.total + weight

    def get_normalize(self):
        temp = 0
        for sentence in self.sentences:
            sentence.weight = sentence.weight / self.total
            temp += sentence.weight

        self.total = temp
