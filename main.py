from nltk.corpus import comtrans
import time
import A
import B

if __name__ == '__main__':
    aligned_sents = comtrans.aligned_sents()[:350]

    startA = time.time()
    A.main(aligned_sents)
    endA = time.time()

    startB = time.time()
    B.main(aligned_sents)
    endB = time.time()

    print "The time for part A is: ", endA-startA
    print "The time for part B is: ", endB-startB
