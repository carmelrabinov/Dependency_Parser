from .chu_liu import Digraph
import numpy as np
import time
import os
import pickle


def load_model(Fn):
    with open(Fn + '\\model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


class DependencyParser:

    def __init__(self):
        self.w = 0
        self.mode = 'base'
        self.logger = {}
        self.features = {}
        self.features_size = 0
        self.curr_sentence = 0
        self.pos = {'#': 44, '$': 33, "''": 28, '(': 36, ')': 37, ',': 1, '.': 10, ':': 30, 'CC': 13,
                     'CD': 2, 'DT': 7, 'EX': 27, 'FW': 40, 'IN': 9, 'JJ': 4, 'JJR': 31, 'JJS': 24, 'LS': 43,
                      'MD': 5, 'NN': 8, 'NNP': 0, 'NNPS': 34, 'NNS': 3, 'PDT': 38, 'POS': 25, 'PRP': 18,
                       'PRP$': 23, 'RB': 16, 'RBR': 19, 'RBS': 39, 'RP': 22, 'SYM': 42, 'TO': 17,
                        'UH': 41, 'VB': 6, 'VBD': 14, 'VBG': 12, 'VBN': 15, 'VBP': 21, 'VBZ': 11,
                         'WDT': 20, 'WP': 29, 'WP$': 35, 'WRB': 32, '``': 26, 'root': 45}
        self.pos_size = 46
        self.vocabulary = {'root': 0}
        self.v_size = 0

    def data_preprocessing(self, path, mode):
        """
        preprocess the data and save it to data and data edges structures
        :param path: path to data file
        :param mode: train mode or test mode
        :return: None
        """
        self.data = []
        self.data_edges = []
        with open(path, 'r') as f:
            counter = 1
            sentence_data = {}
            sentence_edges = {}
            for line in f:
                word = line.split('\t', 10)
                # if line has ended it means the sentence is ended and append it to the data list
                if word[0] == '\n':
                    self.data.append(sentence_data)
                    self.data_edges.append(sentence_edges)
                    sentence_data = {}
                    sentence_edges = {}
                else:
                    # word[0] is the child index
                    # word[1] is the child word
                    # word[3] is the child pos
                    # word[6] is the parent index

                    # add the child word to the vocabulary dictionary
                    if mode == 'train' and word[1] not in self.vocabulary.keys():
                        self.vocabulary[word[1]] = counter
                        counter += 1

                    # add child data to the sentence_data dictionary
                    sentence_data[int(word[0])] = (word[6], word[1], word[3])

                    # add the edge parent_index --> child_index to sentence_edges dictionary
                    if int(word[6]) not in sentence_edges:
                        sentence_edges[int(word[6])] = [int(word[0])]
                    else:
                        sentence_edges[int(word[6])].append(int(word[0]))

        for sentence_edges, sentence in zip(self.data_edges, self.data):
            for i in range(len(sentence)):
                if i not in sentence_edges.keys():
                    sentence_edges[i] = []
            
        if mode == 'train':
            self.v_size = len(self.vocabulary)

    def create_succesors(self, sen_length):
        """
        :param sen_length: sentence length
        :return: all possible edges in a sentence in the form of a dict[parent] = list of children
        """
        succesors = {}
        for i in range(0, sen_length):
            succesors[i] = list(range(1, sen_length + 1))
            if i > 0:
                succesors[i].remove(i)
        return succesors

    def get_score(self, parent_id, child_id):
        """
        return the weight of an edge (w * feature)
        :param parent_id: parent word number in the sentence (root is 0)
        :param child_id: child word number in the sentence
        :return: a scalar - the weight of an edge (w * feature)
        """
        # add feature to feature dictionary if does not exist
        self.update_feature_to_feature_dict(self.curr_sentence, parent_id, child_id)

        # perform dot multiplication
        return np.sum(self.w[self.features[(self.curr_sentence, parent_id, child_id)]])

    def update_feature_to_feature_dict(self, sen_number, parent_id, child_id):
        """
        save the feature for the edge (parent -> child) to the features dictionary if it does not exist
        :param sen_number: sentence number
        :param parent_id: parent word number in the sentence (root is 0)
        :param child_id: child word number in the sentence
        :return: None
        """

        # this method is an auxiliary to get_feature method that ensures that
        # the feature is already calculated and in the feature dictionary
        if (sen_number, parent_id, child_id) in self.features:
            return
        else:
            if parent_id == 0:
                parent_data = ('root', 'root')
            else:
                parent_data = self.data[sen_number][parent_id][1:]
            child_data = self.data[sen_number][child_id][1:]
            edge_feature = self.get_features(parent_data, child_data)
            self.features[(sen_number, parent_id, child_id)] = edge_feature
            return

    def get_glm(self, mst):
        """
        :param mst: a minimum spanning tree of a sentence in the form of a dict[parent] = list of children
        :return: sum of all edges feature as a list of indexes
        """
        glm = []
        # iterate over all edges in the mst
        for parent, children_list in mst.items():
            for child in children_list:

                # add feature to feature dictionary if does not exist
                self.update_feature_to_feature_dict(self.curr_sentence, parent, child)

                # add feature indexes to the glm
                glm.extend(self.features[(self.curr_sentence, parent, child)])

        return glm

    # def predict_sentence(self, sentence):
    #     graph = Digraph(self.create_succesors(len(sentence)), get_score=self.get_score)
    #     result = graph.mst().successors

    def test(self, test_path):
        """
        predict and test accuracy of test set
        :param test_path: path to test set data file
        :return: accuracy in percentage over the test set
        """

        t_start = time.time()
        self.data_preprocessing(test_path, mode='test')
        correct_counter = 0
        total_counter = 0

        self.logger['test data path'] = test_path
        self.logger['test sentences number'] = len(self.data)

        # iterate ove all the data
        for sen_idx, sen in enumerate(self.data):
            self.curr_sentence = sen_idx

            # predict the mst
            graph = Digraph(self.create_succesors(len(sen)), get_score=self.get_score)
            mst = graph.mst().successors

            # compare the predicted mst to the true one and count all correct edges
            for i in range(len(mst)):
                correct_counter += len([w for w in mst[i] if w in self.data_edges[sen_idx][i]])
                total_counter += len(self.data_edges[sen_idx][i])

        # clean the data
        del self.data, self.data_edges
        self.features = {}

        accuracy = np.float(correct_counter) / np.float(total_counter) * 100
        print('finished testing in {} minutes, test accuracy: {}'.format((time.time() - t_start)/60, accuracy))
        self.logger['test accuracy'] = accuracy
        self.logger['test time [minutes]'] = (time.time() - t_start) / 60
        return accuracy

    def train(self, data_path, max_iter=20, mode='base'):
        """
        train weights given a data set of sentences
        :param data_path: path to train set data file
        :param max_iter: max number of iteration that will be performed in the perceptron algorithm
        :param mode: base mode or complex mode of features
        :return:
        """
        self.mode = mode
        self.data_preprocessing(data_path, mode='train')
        self.features_size = self.get_features_size()
        self.w = np.zeros(self.features_size)

        self.logger['mode'] = mode
        self.logger['vocabulary size'] = self.v_size
        self.logger['features size'] = self.features_size
        self.logger['pos size'] = self.pos_size
        self.logger['train data path'] = data_path
        self.logger['train sentences number'] = len(self.data)
        self.logger['max iterations'] = max_iter

        t_start = time.time()
        # run perceptron algorithm for max_iter times
        for i in range(max_iter):

            early_stop_conditions = True

            for sen_idx, sen in enumerate(self.data):
                self.curr_sentence = sen_idx
                graph = Digraph(self.create_succesors(len(sen)), get_score=self.get_score)
                result = graph.mst().successors
                # if predicted mst is different from the true
                if result != self.data_edges[sen_idx]:
                    early_stop_conditions = False
                    predicted_feature = self.get_glm(result)
                    true_feature = self.get_glm(self.data_edges[sen_idx])
                    np.add.at(self.w, true_feature, 1)
                    np.add.at(self.w, predicted_feature, -1)

            print('finished iteration {} in {} minutes'.format(i+1, (time.time() - t_start)/60))

            # if all predictions are correct - stop the training
            if early_stop_conditions:
                break

        self.logger['training time [minutes]'] = (time.time() - t_start)/60

        # clean the data
        del self.data, self.data_edges
        self.features = {}

    def get_features(self, parent_node, child_node, get_size=False):
        """
        create a feature vector as list of indexes of the ones for a directed edge(parent_node --> child_node)
        :param parent_node: all parent data (wod, pos, etc) as a tuple
        :param child_node: all child data (wod, pos, etc) as a tuple
        :param get_size: if True return only the feature vector size
        :return: feature vector as list of indexes
        """

        (parent_word, parent_pos) = parent_node
        (child_word, child_pos) = child_node

        try:
            parent_word_ind = self.vocabulary[parent_word]
        except:
            parent_word_ind = -1
        parent_pos_ind = self.pos[parent_pos]
        try:
            child_word_ind = self.vocabulary[child_word]
        except:
            child_word_ind = -1
        child_pos_ind = self.pos[child_pos]

        curr_size = 0
        feature = []

        ### Base Features:
        # feature 1: p-word, p-pos
        f1 = curr_size + parent_word_ind*self.pos_size + parent_pos_ind
        curr_size += self.pos_size*self.v_size   # size: POS size * Vocabulary size
        if parent_word_ind != -1:
            feature.append(f1)

        # feature 2: p-word
        f2 = curr_size + parent_word_ind
        curr_size += self.v_size    # size: Vocabulary size
        if parent_word_ind != -1:
            feature.append(f2)

        # feature 3: p-pos
        f3 = curr_size + parent_pos_ind
        curr_size += self.pos_size    # size: POS size
        feature.append(f3)

        # feature 4: c-word, c-pos
        f4 = curr_size + child_word_ind*self.pos_size+child_pos_ind
        curr_size += self.pos_size*self.v_size  # size = POS size * Vocabulary size
        if child_word_ind != -1:
            feature.append(f4)

        # feature 5: c-word
        f5 = curr_size + child_word_ind  # size: Vocabulary size
        curr_size += self.v_size  # size: Vocabulary size
        if parent_word_ind != -1:
            feature.append(f5)

        # feature 6: p-pos
        f6 = curr_size + child_pos_ind
        curr_size += self.pos_size  # size: POS size
        feature.append(f6)

        # feature 8: p-pos, c-pos, c-word
        f8 = curr_size + child_word_ind*(self.pos_size**2) + child_pos_ind*self.pos_size + parent_pos_ind
        curr_size += self.pos_size**2 * self.v_size    # size: (POS size)^2 * Vocabulary size
        if child_word_ind != -1:
            feature.append(f8)

        # feature 10: p-word, p-pos, c-pos
        f10 = curr_size + parent_word_ind*(self.pos_size**2) + child_pos_ind*self.pos_size+parent_pos_ind
        curr_size += self.pos_size**2 * self.v_size    # size: (POS size)^2 * Vocabulary size
        if parent_word_ind != -1:
            feature.append(f10)

        # feature 13: p-pos, c-pos
        f13 = curr_size + parent_pos_ind*self.pos_size+child_pos_ind
        curr_size += self.pos_size ** 2  # size: POS size^2
        feature.append(f13)

        ### Complex Features:
        # if self.mode == 'complex':

        if get_size:
            return curr_size

        return feature

    def get_features_size(self):
        """
        :return: feature vector size
        """
        return self.get_features(('root', 'root'), ('root', 'root'), get_size=True)
    
    def save_model(self, resultsfn):
        """
        saves the parser as a pickle file
        :param resultsfn: path to directory to save in the pickle file
        :return: None
        """
        print('Saving model to {}'.format(resultsfn))
        # creating directory
        if not os.path.exists(resultsfn):
            os.makedirs(resultsfn)
        # dump all results:
        with open(resultsfn + '\\model.pkl', 'wb') as f:
            pickle.dump(self, f)

    def print_logs(self, resultsfn):
        """
        saves the logs
        :param resultsfn: path to directory to save in the log file
        :return: None
        """
        print('Saving logs to {}'.format(resultsfn))
        with open(resultsfn + '\\logs.txt', 'w') as f:
            for key, value in sorted(self.logger.items()):
                f.write('{}: {}\n'.format(key, value))