"""Graph2Vec module."""

import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import logging
import coloredlogs
coloredlogs.install()


class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + \
                sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + \
            list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


class Graph2vec:
    def __init__(
        self,
        dimensions=64,
        epochs=100,
        min_count=5,
        wl_iterations=3,
        learning_rate=0.025,
        down_sampling=0.0001
    ):
        self.dimensions = dimensions
        self.epochs = epochs
        self.min_count = min_count
        self.wl_iterations = wl_iterations
        self.learning_rate = learning_rate
        self.down_sampling = down_sampling
        self.workers = cpu_count()

    def dataset_reader(self, path):
        """
        Function to read the graph and features from a json file.
        :param path: The path to the graph json.
        :return graph: The graph object.
        :return features: Features hash table.
        :return name: Name of the graph.
        """
        name = path.split('.')[0].split('/')[-1]
        data = json.load(open(path))
        graph = nx.from_edgelist(data["edges"])

        if "features" in data.keys():
            features = data["features"]
        else:
            features = nx.degree(graph)

        features = {int(k): v for k, v in features}
        return graph, features, name

    def feature_extractor(self, path, rounds):
        """
        Function to extract WL features from a graph.
        :param path: The path to the graph json.
        :param rounds: Number of WL iterations.
        :return doc: Document collection object.
        """
        graph, features, name = self.dataset_reader(path)
        machine = WeisfeilerLehmanMachine(graph, features, rounds)
        doc = TaggedDocument(
            words=machine.extracted_features, tags=["g_" + name])
        return doc

    def train(self, input_paths):
        """
        Main function to read the graph list, extract features.
        Learn the embedding and save it.
        :param args: Object with the arguments.
        """
        logging.info("Training Graph2vec...")
        graphs = input_paths
        document_collections = Parallel(n_jobs=self.workers, prefer="threads")(
            delayed(self.feature_extractor)(g, self.wl_iterations) for g in tqdm(graphs, desc='Load data'))

        self.model = Doc2Vec(document_collections,
                             vector_size=self.dimensions,
                             window=0,
                             min_count=self.min_count,
                             dm=0,
                             sample=self.down_sampling,
                             workers=self.workers,
                             epochs=self.epochs,
                             alpha=self.learning_rate)

        out = []
        for f in graphs:
            identifier = f.split('.')[0].split('/')[-1]
            out.append([str(identifier)] +
                       list(self.model.docvecs["g_"+identifier]))
        column_names = ["type"]+["x_"+str(dim)
                                 for dim in range(self.dimensions)]
        out = pd.DataFrame(out, columns=column_names)
        return out

    def apply(self, input_paths):
        logging.info('Applying Graph2vec...')
        graphs = input_paths
        document_collections = Parallel(n_jobs=self.workers, prefer="threads")(
            delayed(self.feature_extractor)(g, self.wl_iterations) for g in tqdm(graphs, desc='Load data'))

        out = []
        for f, doc in tqdm(zip(graphs, document_collections), total=len(graphs), desc='Get embeddings'):
            identifier = f.split('.')[0].split('/')[-1]
            out.append([str(identifier)] +
                       list(self.model.infer_vector(doc.words)))
        column_names = ["type"]+["x_"+str(dim)
                                 for dim in range(self.dimensions)]
        out = pd.DataFrame(out, columns=column_names)
        return out

    def save_model(self):
        self.model.save('model/graph2vec.model')

    def load_model(self, file_path):
        self.model = Doc2Vec.load(file_path)
