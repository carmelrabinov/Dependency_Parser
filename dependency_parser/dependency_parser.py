from .chu_liu import Digraph
import numpy as np



class DependencyParser:
    
    def __init__(self):
        self.w = 0
        self.features_size = 0
        self.curr_sentence = 0
        self.pos = {'#': 44, '$': 33, "''": 28, '(': 36, ')': 37, ',': 1, '.': 10, ':': 30, 'CC': 13,
                     'CD': 2, 'DT': 7, 'EX': 27, 'FW': 40, 'IN': 9, 'JJ': 4, 'JJR': 31, 'JJS': 24, 'LS': 43,
                      'MD': 5, 'NN': 8, 'NNP': 0, 'NNPS': 34, 'NNS': 3, 'PDT': 38, 'POS': 25, 'PRP': 18,
                       'PRP$': 23, 'RB': 16, 'RBR': 19, 'RBS': 39, 'RP': 22, 'SYM': 42, 'TO': 17,
                        'UH': 41, 'VB': 6, 'VBD': 14, 'VBG': 12, 'VBN': 15, 'VBP': 21, 'VBZ': 11,
                         'WDT': 20, 'WP': 29, 'WP$': 35, 'WRB': 32, '``': 26, 'root': 45}

    def data_preprocessing(self, path):
        data = []
        data_edges = []
        with open(path, 'r') as f:
            sentence_data = {}
            sentence_edges = {}
            for line in f:
                word = line.split('\t', 10)
                if word[0] == '\n':
                    data.append(sentence_data)
                    data_edges.append(sentence_edges)
                    sentence_data = {}
                    sentence_edges = {}
                else:
                    sentence_data[int(word[0])] = (word[6], word[1], word[3])
                    if word[6] not in sentence_edges:
                        sentence_edges[int(word[6])] = [int(word[0])]
                    else:
                        sentence_edges[int(word[6])].append(int(word[0]))
        for sentence_edges, sentence in zip(data_edges, data):
            for i in range(len(sentence)):
                if i not in sentence_edges.keys():
                    sentence_edges[i] = []
            
        self.data = data
        self.data_edges = data_edges


#    def predict(self, sentence):
          
    def create_succesors(self, sen_length):
        succesors = {}
        for i in range(0, sen_length+1):
            succesors[i] = list(range(1, sen_length+1))
            if i > 0:
                succesors[i].remove(i)
        return succesors

    
    def get_score(self, node_id_1, node_id_2):
        # direction is: node_id_1 --> node_id_2       
        if node_id_1 == 0:
            parent_data = ('root', 'root')
        else:
            parent_data = self.data[self.curr_sentence][node_id_1][1:]
        child_data = self.data[self.curr_sentence][node_id_2][1:]
        edge_feature = self.get_features(parent_data, child_data)
        return np.dot(edge_feature, self.w)
    
    def train(self, data_path):
        self.data_preprocessing(data_path)
        self.features_size = self.get_features_size()
        self.w = np.zeros(self.features_size)
        for sen_idx, sen in enumerate(self.data):
            self.curr_sentence = sen_idx
            graph = Digraph(self.create_succesors(len(sen)), get_score=self.get_score)
            result = graph.mst().successors
            if result != self.data_edges[sen_idx]:
                predicted_feature = self.get_glm(result)
                true_feature = self.get_glm(self.data_edges[sen_idx])
                self.w += true_feature - predicted_feature

    def get_features(self, parent_node, child_node):
        # direction is: node_id_1 --> node_id_2
        # for now: node = (word, pos), or 'root'
        feature = np.zeros(self.features_size)
        (parent_word, parent_pos) = parent_node
        (child_word, child_pos) = child_node
        
        # feature 3
        feature[self.pos[parent_pos]] = 1
        
        return feature

    def get_features_size(self):
        return len(self.pos)
    
    def get_glm(self, mst):
        glm = np.zeros(self.features_size)
        for parent, children_list,  in mst.items():
            # if len(children_list) < 2:
            #     children_list = [children_list]
            for child in children_list:
                if parent == 0:
                    parent_data = ('root', 'root')
                else:
                    parent_data = self.data[self.curr_sentence][parent][1:]
                child_data = self.data[self.curr_sentence][child][1:]
                glm += self.get_features(parent_data, child_data)
        return glm

'''
train:
1. get sentence, extract his tree
2. create a succesors dictionary including all possible edges for the sentence
3. using cio_lau(W) get best MST 
4. if sentence tree != MST update weights by:
    5. get the MST, sum all get_score pairs to get f1
    6. get tree, sum all get_score pairs to get f2
    7. w += f2 - f1
    
    
predict (w):
1. get sentence
2. create a succesors dictionary including all possible edges for the sentence
3. return ciu_lau(w) results


get_score(w, node_id_1, node_id_2):
    # direction is: node_id_1 --> node_id_2
    f = get features of (node[node_id_1], node[node_id_2])
    return f.dot(w)
    # node is the data relevant like (word[i], pos[i], pos[i-1] etc...)

for more efficient training we can calculate all posible pairs features and then
get_score(w, node_id_1, node_id_2):
    # direction is: node_id_1 --> node_id_2
    return f_dict[(node_id_1, node_id_2)].dot(w)

'''