import config
from astnode import ASTNode
MAX_N_GRAMS = 4

def collect_ngrams(train_sentence_index,Y_train,alignments,unedited_words):
    ngrams = [[]]
    current_ngrams = [[]]
    for i in range(1,MAX_N_GRAMS+1):
        ngrams.append([[]])
        current_ngrams.append(None)
    current_ngram_depth=0
    aux_collect_ngrams(Y_train,alignments,unedited_words,ngrams,current_ngrams,current_ngram_depth)

def aux_collect_ngrams(train_sentence_index,node,alignments,unedited_words,ngrams,current_ngrams,current_ngram_depth):
    # test alignment
    target_w = extract_word(train_sentence_index,node)
    source_w = alignments(target_w)
    if source_w in unedited_words:
        current_ngram_depth=min(current_ngram_depth+1,MAX_N_GRAMS)
        for i in range(current_ngram_depth,0):
            current_ngrams[i] = current_ngrams[i-1]+[target_w]
            ngrams[i].append(current_ngrams[i])
    else:
        current_ngram_depth=0
        for i in range(1,MAX_N_GRAMS+1):
            current_ngrams[i]=[]

    for child in node.children:
        aux_collect_ngrams(child,alignments,unedited_words,ngrams,current_ngrams,current_ngram_depth)

def extract_word(train_sentence_index,node):
    # TODO
