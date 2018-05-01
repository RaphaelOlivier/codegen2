import config
import copy
import simi
from nn.utils.io_utils import deserialize_from_file
from astnode import ASTNode

import numpy as np
import re

MAX_N_GRAMS = 4
MAX_RETRIEVED_SENTENCES = 10

APPLY_RULE = 0
GEN_TOKEN = 1
COPY_TOKEN = 2
GEN_COPY_TOKEN = 3

ACTION_NAMES = {APPLY_RULE: 'APPLY_RULE',
                GEN_TOKEN: 'GEN_TOKEN',
                COPY_TOKEN: 'COPY_TOKEN',
                GEN_COPY_TOKEN: 'GEN_COPY_TOKEN'}

# helper copied from lang.py.py_dataset.py - would NOT work with django


def get_terminal_tokens(_terminal_str):
    """
    get terminal tokens
    break words like MinionCards into [Minion, Cards]
    """
    tmp_terminal_tokens = [t for t in _terminal_str.split(' ') if len(t) > 0]
    _terminal_tokens = []
    for token in tmp_terminal_tokens:
        sub_tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', token).split(' ')
        _terminal_tokens.extend(sub_tokens)

        _terminal_tokens.append(' ')

    return _terminal_tokens[:-1]


def collect_ngrams(aligned_entry, entry_index, act_sequence, unedited_words, simi_score, s2s_alignment_dict):
    current_ngrams = [[]]
    ngrams = [[]]
    for i in range(1, MAX_N_GRAMS+1):
        current_ngrams.append(None)
        ngrams.append([])
    current_ngram_depth = 0
    init_timestep = 0
    node = aligned_entry.parse_tree
    actions = aligned_entry.actions
    alignments = alter_for_copy(np.copy(aligned_entry.alignments), s2s_alignment_dict)
    assert(len(alignments) == len(actions))
    final_timestep = aux_collect_ngrams(entry_index, actions, act_sequence, node, alignments, unedited_words, simi_score,
                                        ngrams, current_ngrams, current_ngram_depth, init_timestep)
    # print(final_timestep, len(actions)-1)  # not including the eos action
    assert(final_timestep == len(actions))
    return ngrams


def aux_collect_ngrams(entry_index, actions, act_sequence, node, alignments, unedited_words, simi_score, ngrams, current_ngrams, current_ngram_depth, timestep):
    # test alignment
    target_w = Gram(entry_index, actions[timestep], act_sequence[timestep], simi_score)
    source_w = alignments[timestep]
    if source_w in unedited_words.values():
        current_ngram_depth = min(current_ngram_depth+1, MAX_N_GRAMS)
        for i in range(current_ngram_depth, 0, -1):
            current_ngrams[i] = current_ngrams[i-1]+[target_w]
            ngrams[i].append(current_ngrams[i])
    else:
        current_ngram_depth = 0
        for i in range(1, MAX_N_GRAMS+1):
            current_ngrams[i] = []
    timestep += 1
    if isinstance(node, ASTNode):
        if node.children:
            for child in node.children:
                timestep = aux_collect_ngrams(entry_index, actions, act_sequence, child, alignments, unedited_words, simi_score, ngrams,
                                              copy.deepcopy(current_ngrams), current_ngram_depth, timestep)
        elif node.value is not None:
            terminal_tokens = get_terminal_tokens(str(node.value))
            for tk in terminal_tokens:
                timestep = aux_collect_ngrams(entry_index, actions, act_sequence, tk, alignments, unedited_words, simi_score, ngrams,
                                              current_ngrams, current_ngram_depth, timestep)

    return timestep


class Gram:
    def __init__(self, entry_index, action, act_ids, score):
        self.entry_index = entry_index
        self.action_type = ACTION_NAMES[action.act_type]
        self.rule_id = None
        self.token_id = None
        self.copy_id = None

        if action.act_type == APPLY_RULE:
            self.rule_id = act_ids[0]

        elif action.act_type == GEN_TOKEN:
            self.token_id = act_ids[1]

        elif action.act_type == COPY_TOKEN:
            self.copy_id = act_ids[2]

        else:
            assert(action.act_type == GEN_COPY_TOKEN)
            self.token_id = act_ids[1]
            self.copy_id = act_ids[2]

        self.score = score

    def __repr__(self):

        if self.action_type == "APPLY_RULE":
            return str((self.action_type, self.rule_id))

        elif self.action_type == "GEN_TOKEN":
            return str((self.action_type, self.token_id))

        elif self.action_type == "COPY_TOKEN":
            return str((self.action_type, self.copy_id))

        else:
            assert(self.action_type == "GEN_COPY_TOKEN")
            return str((self.action_type, self.token_id, self.copy_id))

    def equals(self, ng):
        return self.action_type == ng.action_type and self.rule_id == ng.rule_id and self.copy_id == ng.copy_id and self.token_id == ng.token_id


def alter_for_copy(alignments, s2s_alignment_dict):
    for t in range(alignments.shape[0]):
        new_index = s2s_alignment_dict[alignments[t]][3]
        if new_index is not None:
            alignments[t] = new_index
    return alignments


def insert_ngram(ng, ngram_list):
    for i, ng2 in enumerate(ngram_list):
        eq = True
        for j in range(len(ng)):
            if not ng[j].equals(ng2[j]):
                eq = False
                break
        if eq:
            if ng[0].score > ng2[0].score:
                ngram_list[i] = ng
            return
    ngram_list.append(ng)


def retrieve_translation_pieces(dataset, input_sentence):

    all_ngrams = [[] for k in range(MAX_N_GRAMS+1)]
    simi_scores = []
    for entry in dataset.examples:
        simi_scores.append(simi.simi(input_sentence, entry.query, True))
    top_indices = np.argsort(np.array(simi_scores))[-MAX_RETRIEVED_SENTENCES:][::-1]
    for i in top_indices:

        matrix, dist = simi.sentence_distance(input_sentence, dataset.examples[i].query, True)
        unedited_words, first_index_dict, second_index_dict = simi.align(
            input_sentence, dataset.examples[i].query, matrix, False, True)
        act_sequence = dataset.data_matrix["tgt_action_seq"][i]
        ngrams = collect_ngrams(dataset.examples[i], i, act_sequence,
                                unedited_words, simi_scores[i], second_index_dict)

        for i in range(1, MAX_N_GRAMS+1):
            all_ngrams[i] += ngrams[i]
    print[len(q) for q in all_ngrams[1:]]
    max_ngrams = [[] for k in range(MAX_N_GRAMS+1)]
    for i in range(1, MAX_N_GRAMS+1):
        for ng in all_ngrams[i]:
            insert_ngram(ng, max_ngrams[i])
    print[len(q) for q in max_ngrams[1:]]
    print(input_sentence)
    return max_ngrams


if __name__ == "__main__":
    train_data, dev_data, test_data = deserialize_from_file('../../files/aligned_hs.bin')
    input_sentence = test_data.examples[10].query
    retrieve_translation_pieces(train_data, input_sentence)
