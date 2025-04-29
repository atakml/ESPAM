class Explainer:
    def __init__(self):
        self.model = None
        self.activation_patterns = None
        self.graph_dataset = None
        self.activation_matrix = None
        self.probabilist_model = None
        self.rules = None

    def beam_search(self):
        pass

    def compute_probabilistic_model_from_activation_matrix(self):
        pass

    def compute_activation_matrix_from_embeddings(self, embeddings):
        pass

    def compute_metrics_for_rules(self):
        pass

    def compute_ot_from_rule(self, rule):
        pass

    def load_model(self):
        pass

    def load_dataset(self):
        pass
