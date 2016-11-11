import json
import numpy as np
import collections
import nltk

class Loader:
    def __init__(self):
        self.industries = dict()
    
    def tokenize(self, text):
        text = text.rstrip()
        return nltk.word_tokenize(text)
        
    def build_vocab(self, data):
        data_str = ' '.join(data)
        data_list = data_str.split(' ')
        counter = collections.Counter(data_list)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))
        return word_to_id
            
    def padding(self, desc, forced_sequence_length, padding_word = "PAD"):
        desc = self.tokenize(desc)
        num_padding = forced_sequence_length - len(desc)
        if (num_padding > 0):
            padded = [padding_word] *  num_padding + desc
        else:
            padded = desc[0:forced_sequence_length]
        return padded
       
    def load_embeddings(self, embedding_size):
        word_embeddings = {}
        for word in self.word_to_id:
            word_embeddings[word] = np.random.uniform(-0.25, 0.25, embedding_size)
        return word_embeddings
       
    def invert_vocab(self, vocab):
        inv_vocab = {v: k for k, v in vocab.items()}
        return inv_vocab
        
    def load_data(self, filename):
        self.data = []
        self.category = []
        with open(filename) as data_file:
            counter_max = 10000
            for line in data_file:
                if (counter_max > 0):
                    current_example = json.loads(line)
                    self.data.append(' '.join(self.padding(current_example['description'], forced_sequence_length = 405)))
                    self.category.append(current_example['industry'])
                    counter_max = counter_max - 1
                else:
                    break
        labels = sorted(list(set(self.category))) 
        num_labels = len(labels)
        one_hot = np.zeros((num_labels, num_labels), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))
        y_raw = []
        for c in self.category:
            y_raw.append(label_dict[c])
        
        x_raw = []         
        for desc in self.data:
            x_raw.append(desc.split())
#        x_raw = self.pad_description(self.data, forced_sequence_length = 405)
        self.word_to_id = self.build_vocab(self.data)
        x = np.array([[self.word_to_id[word] for word in sentence] for sentence in x_raw])
        y = np.array(y_raw)
        return x, y, self.word_to_id, labels
        
    
#    def batch_iter(data, batch_size, num_epochs, shuffle=True):
#        data = np.array(data)
#        data_size = len(data)
#        num_batches_per_epoch = int(data_size / batch_size) + 1
#
#        for epoch in range(num_epochs):
#            if shuffle:
#                shuffle_indices = np.random.permutation(np.arange(data_size))
#                shuffled_data = data[shuffle_indices]
#            else:
#                shuffled_data = data
#
#            for batch_num in range(num_batches_per_epoch):
#                start_index = batch_num * batch_size
#                end_index = min((batch_num + 1) * batch_size, data_size)
#                yield shuffled_data[start_index:end_index]
#        
            
    
    
if __name__ == "__main__":
    train_file = 'data/companies.jsons'
    my_loader = Loader()
    my_loader.load_data(train_file)