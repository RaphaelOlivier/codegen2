# assuming requirements: output layer, Lx, Dx
class UnigramCreator:
	def __init__(self):
    	self.Lx = []

    def create_unigrams(Dx):
    	for key in Dx:
    		unigram_list = key.split(" ")
    		for word in unigram_list:
    			if word not in self.Lx:
    				self.Lx.append(word)
# create Lx



unigram = UnigramCreator()
Lx =  unigram.create_unigrams(Dx)


def get_ngram(begin, end, word, predicted_terminal_nodes):

	index = predicted_terminal_nodes.index()
	n_gram = ''
	n_gram_selector = begin - end
	if abs(n_gram_selector) == 0:
		n_gram = predicted_terminal_nodes[index-1] + ' ' + predicted_terminal_nodes[index]
	elif abs(n_gram_selector) == 1:
		n_gram = predicted_terminal_nodes[index-2] + ' ' + predicted_terminal_nodes[index-1] + ' ' + predicted_terminal_nodes[index]
	elif abs(n_gram_selector) == 2:
		n_gram = predicted_terminal_nodes[index-3] + ' ' + predicted_terminal_nodes[index-2] + ' ' + predicted_terminal_nodes[index-1] + predicted_terminal_nodes[index]
	return n_gram 

# call this for each time step , instead of going through the whole vocabulary, we go through only unigrams
def NMTPredictionUpdation(score, timestep, predicted_terminal_nodes):
	lda = 
	for u in Lx:
		score[u] = score[u] + lbda * Dx[u]
		for i in range(1,3):
			if timestep - i < 1:
				break
			n_gram = get_ngram(timestep - i , timestep -1, u, predicted_terminal_nodes)
			if n_gram not in Dx:
				break
			score[u] = score[u] + lda * Dx[n_gram]

	return score

  

