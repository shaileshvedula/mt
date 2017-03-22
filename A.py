import nltk
import time
from nltk.align import IBMModel1
from nltk.align import IBMModel2
from nltk.align import AlignedSent, Alignment

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
    ibm1 = IBMModel1(aligned_sents, 10)
    return ibm1

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    ibm2 = IBMModel2(aligned_sents, 10)
    return ibm2

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    count = 0
    temp =[]    
    for i in range(n):
        aligned = model.align(aligned_sents[i])
        #print aligned_sents[i].words
        #print aligned_sents[i].mots
        #print aligned.alignment_error_rate(aligned_sents[i])
        #print '\n'
        #temp.append([aligned_sents[i].words, aligned_sents[i].mots, aligned.alignment_error_rate(aligned_sents[i])])
        count += aligned.alignment_error_rate(aligned_sents[i])
    return float(count)/n
            
# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    
    fileout = open(file_name, 'w')
    for i in range(20):
        output = model.align(aligned_sents[i])
        #print output.mots, type(output.mots)
        fileout.write(str(output.mots) + '\n')
        fileout.write(str(output.words) + '\n')

        string = ['{0}-{1}'.format(x[0], x[1]) for x in output.alignment]
        fileout.write(' '.join(string) + '\n')

        fileout.write('\n')
    fileout.close()
    

def main(aligned_sents):
    #start = time.time()
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer= compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
   # end = time.time()

   # print "The time take is: ", end-start

    #for i, elem in enumerate(list1):
    #    if elem[2] > list2[i][2]:
    #        print 'IBM1 greater than IBM2 for sentence:'
    #        print elem[0]
    #        print elem[1]
    #        print 'IBM1 score {0}, IBM2 score {1}'.format(elem[2], list2[i][2])
    #        print '\n'
    #    elif elem[2] < list2[i][2]:

    #        print 'IBM2 greater than IBM1 for sentence:'
    #        print elem[0]
    #        print elem[1]
    #        print 'IBM1 score {0}, IBM2 score {1}'.format(elem[2], list2[i][2])
    #        print '\n' 
    










