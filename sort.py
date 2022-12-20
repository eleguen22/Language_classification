import pickle
from model_tree import ModelTree
from model_adaboost import AdaModel
import sys


def predict(model_file, test_file=None):
    model_file = open(model_file, "rb")
    decision_model = pickle.load(model_file)
    model_file.close()

    decision_model.prediction(test_file)


def train(sentences, model_file, learning_type):
    if learning_type == "ada":
        model = AdaModel(train_file=sentences, model_file=model_file)
        model.train_ada()
    else:
        model = ModelTree(train_file=sentences, model_file=model_file)
        model.train_tree()


def main():
    command = sys.argv[1]

    if command == "predict":
        model_file = sys.argv[2]
        if sys.argv[3] is None:
            test_file = None
        else:
            test_file = sys.argv[3]
        predict(model_file, test_file)

    if command == "train":
        sentences = sys.argv[2]
        model_file=sys.argv[3]
        learning_type = sys.argv[4]
        train(sentences, model_file, learning_type)


if __name__ == '__main__':
    main()
