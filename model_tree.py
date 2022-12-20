from decision_tree import decision_tree
from sentence import parse_file
import pickle


class ModelTree:

    def __init__(self, train_file="./data_in/train_dataset.dat", test_file="./data_in/test_dataset.dat", model_file="./model_out/model_tree.oj"):
        sentences = parse_file((train_file, test_file))
        self.dataset = {"Train": sentences[0], "Test": sentences[1]}
        self.model_file = model_file
        self.tree = None

    def train_tree(self):
        sentences = self.dataset["Train"]
        features = set(sentences[0].features.keys())
        self.tree = decision_tree(sentences, features,[], 10)
        file = open(self.model_file, "wb")
        pickle.dump(self, file)
        file.close()

    def prediction(self, data_file=None):
        if not self.tree :
            self.train_tree()

        if not data_file :
            sentences = self.dataset["Test"]
        else :
            sentences = parse_file([data_file],inference=True)[0]
        result = []

        for sentence in sentences:
            prediction = self.tree.classification(sentence)
            result.append(prediction)

        print(*result, sep='\n')
