from weighted_sentences import WeightedSentences
from sentence import parse_file
from decision_tree import decision_tree
import math
import pickle


class AdaModel:

    def __init__(self, train_file="./data_in/train_dataset.dat", test_file="./data_in/test_dataset.dat", model_file="./model_out/model_tree.oj"):

        sentences = parse_file((train_file, test_file))
        self.dataset = {"Train": sentences[0], "Test": sentences[1]}
        self.model_file = model_file
        self.tree = None
        self.hypothesis_set = []

    def train_ada(self, hypothesis_size=6):
        epsilon = 0.001
        sentences = self.dataset["Train"]
        features = set(sentences[0].features.keys())
        weights_tab = WeightedSentences(sentences)
        self.hypothesis_set = []

        for i in range(hypothesis_size):
            stump = decision_tree(sentences, features,[], 1)
            error = 0

            for sentence in sentences:
                prediction = stump.classification(sentence)
                if prediction != sentence.goal:
                    error += sentence.weight

            if error == 0:
                stump.weight = math.log(((1 - error + epsilon) / (error + epsilon)))
                self.hypothesis_set.append(stump)
                continue

            error = min(error, 1 - epsilon)

            for j in range(len(sentences)):
                sentence = sentences[j]
                prediction = stump.classification(sentence)

                if prediction == sentence.goal:
                    n_weight = sentence.weight * error / (1 - error)
                    weights_tab.change_weight(j, n_weight)

            weights_tab.get_normalize()
            stump.weight = math.log((1 - error) / error)
            self.hypothesis_set.append(stump)

        file = open(self.model_file, "wb")
        pickle.dump(self, file)
        file.close()

    def prediction(self, data_file=None):

        if not self.tree:
            self.train_ada()

        if not data_file:
            sentences = self.dataset["Test"]

        else :
            sentences = parse_file([data_file], inference=True)[0]

        result = []

        for sentence in sentences:
            prediction = self.classify(sentence)
            result.append(prediction)

        print(*result, sep='\n')

    def classify(self, instance):

        result = None
        count_tab = {}
        max_count = 0

        for stump in self.hypothesis_set:
            prediction = stump.classification(instance)

            if prediction in count_tab:
                count_tab[prediction] += stump.weight
            else:
                count_tab[prediction] = stump.weight

            if count_tab[prediction] > max_count:
                max_count = count_tab[prediction]
                result = prediction

        return result
