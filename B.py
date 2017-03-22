import nltk
import A
from nltk.align import AlignedSent, Alignment

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):

        alignment_final = []
        m = len(align_sent.mots)
        l = len(align_sent.words)

        for i, source in enumerate(align_sent.mots):
            final_prob = -float("inf")
            for j, target in enumerate(align_sent.words):
                prob_align = (self.t[(source, target)])*self.q[(j,i,l,m)]
                if prob_align >= final_prob:
                    final_prob = prob_align
                    alignment_point = j
            alignment_final.append((alignment_point, i))
        alignment_final = Alignment(alignment_final)

        return AlignedSent(align_sent.words, align_sent.mots, alignment_final)
            
    
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
        
        t = {}
        q = {}
        nw = {}
        nm = {}
        q = {}
        c = {}

        for sent in aligned_sents:
            source = sent.words
            target = sent.mots
            source_len = len(source)
            target_len = len(target)

            for word in source:
                nw[word] = nw.get(word,0.0) + source_len
            for mot in target:
                nm[mot] = nm.get(mot,0.0) + target_len

        for sent in aligned_sents:
            for e in sent.words:
                for f in sent.mots:
                    t[(f, e)]= 1.0/nw[e]
                    t[(e,f)] = 1.0/nm[f]
            
            m = len(sent.mots)
            l = len(sent.words)

            for i in range(m):
                for j in range(l):
                    q[(i,j,m,l)] = 1.0/(m+1)
                    q[(j,i,l,m)] = 1.0/(l+1)
       
        for itr in range(num_iters):
            for sent in aligned_sents:
                word_list = sent.words
                mot_list = sent.mots
                m = len(mot_list)
                l = len(word_list)

                for i in range(m):
                    for j in range(l):
                        sum1 = 0
                        sum2 = 0
                        for k in range(l):
                            sum1 += q[(k,i,l,m)]*t[(mot_list[i], word_list[k])]
                            sum2 += q[(i,k,m,l)]*t[(word_list[k], mot_list[i])]

                        delta1 = q[(j,i,l,m)]*t[(mot_list[i], word_list[j])]/sum1
                        delta2 = q[(i,j,m,l)]*t[(word_list[j], mot_list[i])]/sum2
                        delta = (delta1+delta2)/2.0

                        c[(word_list[j], mot_list[i])] = c.get((word_list[j], mot_list[i]),0.0) + delta
                        c[(mot_list[i], word_list[j])] =c.get( (mot_list[i], word_list[j]), 0.0) + delta
                        c[mot_list[i]] = c.get(mot_list[i], 0.0) + delta
                        c[word_list[j]] = c.get(word_list[j], 0.0) + delta
                        c[(j,i,l,m)] = c.get((j,i,l,m), 0.0)  + delta
                        c[(i,j,m,l)] = c.get((i,j,m,l), 0.0) + delta
                        c[(i,l,m)] = c.get((i,l,m), 0.0) + delta
                        c[(j,m,l)] = c.get((j,m,l), 0.0) + delta

            for sent in aligned_sents:
                for word in sent.words:
                    for mot in sent.mots:
                        t[(mot, word)] = c[(word,mot)]/c[word]
                        t[(word, mot)] = c[(mot, word)]/c[mot]
                 
                m = len(sent.mots)
                l = len(sent.words)
                for i in range(m):
                    for j in range(l):
                        q[(i,j,m,l)] = c[(i,j,m,l)]/c[(j,m,l)]
                        q[(j,i,l,m)] = c[(j,i,l,m)]/c[(j,m,l)]

        return t,q
      
def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
