import math


class Node:

    def __init__(self, content, leaf=False):

        self.content = content
        self.leaf = leaf
        self.children = {}
        self.weight = None

    def add_child(self, branch_label, node):
        self.children[branch_label] = node

    def classification(self, sentence):

        node = self
        while node:
            child_branch = sentence.features[node.content]

            if child_branch in node.children:
                node = node.children[child_branch]

            if node.leaf:
                return node.content
        return None


def goals_counter(sentences):

    count_dic = {}

    for sentence in sentences:
        if sentence.weight:
            weight = sentence.weight
        else:
            weight = 1 / len(sentences)

        if sentence.goal in count_dic:
            count_dic[sentence.goal] += weight
        else:
            count_dic[sentence.goal] = weight

    return count_dic


def majority_classification(sentences):

    dic=goals_counter(sentences)

    max_key=max(dic,key=dic.get)
    return(max_key)


def same_goal(sentences):

    for i in range(0, len(sentences) - 1):
        if sentences[i].goal != sentences[i + 1].goal:
            return False
    return True


def split_data(sentences, feature):

    result = {}
    for sentence in sentences:
        branch = sentence.features[feature]
        if branch in result:
            result[branch].append(sentence)
        else:
            result[branch] = [sentence]
    return result


def entropy(sentences):

    etpy = 0
    goals_number = goals_counter(sentences)
    for key in goals_number.keys():
        prob = goals_number[key] / len(sentences)
        etpy += -prob * math.log(prob, 2)

    return etpy


def info_gain(sentences, feature, etpy):

    splits = split_data(sentences, feature)
    keys = splits.keys()
    reminder = 0

    for key in keys:
        group = splits[key]
        reminder += (len(group) / len(sentences)) * entropy(group)

    gain = etpy - reminder

    return gain, splits


def find_maxgain(sentences, features):

    etpy = entropy(sentences)
    maxgain = -1
    children = None
    associated_feature = None

    for feature in features:
        gain, splits = info_gain(sentences, feature, etpy)
        if gain > maxgain:
            maxgain = gain
            associated_feature = feature
            children = splits

    return associated_feature, children


def decision_tree(sentences, features, parents, depth=10):

    if same_goal(sentences):
        return Node(sentences[0].goal, leaf=True)
    feature, splits = find_maxgain(sentences, features)
    tree = Node(feature)

    if depth < 1:
        depth = 1

    for content in splits:
        objs = splits[content]

        if depth == 1:
            subtree = Node(majority_classification(objs), leaf=True)
            tree.add_child(content, subtree)
        else:
            subtree = decision_tree(objs, features-{feature}, sentences, depth - 1)
            tree.add_child(content, subtree)

    return tree
