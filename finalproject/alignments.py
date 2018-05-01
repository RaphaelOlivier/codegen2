from dataset import DataSet, DataEntry
from retrievalmodel import RetrievalModel
import warnings
warnings.filterwarnings("ignore")


class AlignedDataSet(DataSet):
    def __init__(self, dataset, alignments):
        self.annot_vocab = dataset.annot_vocab
        self.terminal_vocab = dataset.terminal_vocab
        self.name = dataset.name
        self.examples = [AlignedEntry(e, alignments[i]) for i, e in enumerate(dataset.examples)]
        self.data_matrix = dataset.data_matrix
        self.grammar = dataset.grammar
        self.alignments = list()


class AlignedEntry(DataEntry):
    def __init__(self, example, alignments=None):
        self.raw_id = example.raw_id
        self.eid = example.eid
        # FIXME: rename to query_token
        self.query = example.query
        self.parse_tree = example.parse_tree
        self.actions = example.actions
        self.code = example.code
        self.meta_data = example.meta_data

        if example.dataset is not None:
            self.dataset = example.dataset

        if hasattr(example, "alignments"):
            self.alignments = example.alignments
        else:
            assert(alignments is not None)

            # print example.actions
            # print example.parse_tree.pretty_print()
            if not (len(alignments) == len(self.actions)):
                print(self.raw_id, len(self.actions), len(alignments))
            self.alignments = alignments

    def copy(self):
        return AlignedEntry(super(AlignedEntry, self).copy())


def compute_alignments(model, dataset):

    alignments = []
    for i in range(dataset.count):
        inputs = dataset.get_prob_func_inputs([i])
        algn = model.align(*inputs)[0][0]
        print algn
        alignments.append(algn)
    print len(alignments)
    new_dataset = AlignedDataSet(dataset, alignments)

    return new_dataset
