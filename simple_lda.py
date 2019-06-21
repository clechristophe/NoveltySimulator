import math
import random
import numpy
import numpy.random
import sys
from progressbar import printProgressBar

FILE_NAME = sys.argv[1]
beta_list = [0.01,0.05,0.1,0.5,0.9,0.99]

for beta_param in beta_list:
	TOPIC_N = 10
	VOCABULARY_SIZE = 10000
	DOC_NUM = 1000
	TERM_PER_DOC = 100
	 
	beta = [beta_param for i in range(VOCABULARY_SIZE)]
	alpha = [0.9 for i in range(TOPIC_N)]
	FILE_NAME += str(beta_param)
	 
	phi = []
	## generate multinomial distribution over words for each topic
	for i in range(TOPIC_N):
		topic =	numpy.random.mtrand.dirichlet(beta, size = 1)
		phi.append(topic)
	## generate words for each document
	output_f = open(FILE_NAME+'.doc','w')
	z_f = open(FILE_NAME+'.z','w')
	theta_f = open(FILE_NAME+'.theta','w')

	printProgressBar(0, DOC_NUM, prefix = 'Progress:', suffix = '', length = 50)


	for i in range(DOC_NUM):
		buffer = {}
		z_buffer = {} 
		theta = [numpy.zeros(TOPIC_N)]
		theta[0][numpy.random.choice(TOPIC_N,1)] = 1 # Randomly associate a document to a topic

		for j in range(TERM_PER_DOC):
			z = numpy.random.multinomial(1,theta[0],size = 1)
			z_assignment = 0
			for k in range(TOPIC_N):
				if z[0][k] == 1:
					break
				z_assignment += 1
			if not z_assignment in z_buffer:
				z_buffer[z_assignment] = 0
			z_buffer[z_assignment] = z_buffer[z_assignment] + 1
			w = numpy.random.multinomial(1,phi[z_assignment][0],size = 1)
			w_assignment = 0
			for k in range(VOCABULARY_SIZE):
				if w[0][k] == 1:
					break
				w_assignment += 1
			if not w_assignment in buffer:
				buffer[w_assignment] = 0
			buffer[w_assignment] = buffer[w_assignment] + 1
		## output
		output_f.write(str(i)+'\t'+str(TERM_PER_DOC)+'\t')
		for word_id, word_count in buffer.items():
			output_f.write(str(word_id)+':'+str(word_count)+' ')
		output_f.write('\n')
		z_f.write(str(i)+'\t'+str(TERM_PER_DOC)+'\t')
		for z_id, z_count in z_buffer.items():
			z_f.write(str(z_id)+':'+str(z_count)+' ')
		z_f.write('\n')
		theta_f.write(str(i)+'\t')
		for k in range(TOPIC_N):
			theta_f.write(str(k)+':'+str(theta[0][k])+' ')
		theta_f.write('\n')
		printProgressBar(i + 1, DOC_NUM, prefix = 'Progress:', suffix = str(i), length = 50)

	z_f.close()
	theta_f.close()
	output_f.close()
	 
	## output phi
	output_f = open(FILE_NAME+'.phi','w')
	for i in range(TOPIC_N):
		output_f.write(str(i)+'\t')
		for j in range(VOCABULARY_SIZE):
			output_f.write(str(j)+':'+str(phi[i][0][j])+' ')
		output_f.write('\n')
	output_f.close()
	 
	## output hyper-parameters
	output_f = open(FILE_NAME+'.hyper','w')
	output_f.write('TOPIC_N:'+str(TOPIC_N)+'\n')
	output_f.write('VOCABULARY_SIZE:'+str(VOCABULARY_SIZE)+'\n')
	output_f.write('DOC_NUM:'+str(DOC_NUM)+'\n')
	output_f.write('TERM_PER_DOC:'+str(TERM_PER_DOC)+'\n')
	output_f.write('alpha:'+str(alpha[0])+'\n')
	output_f.write('beta:'+str(beta[0])+'\n')
	output_f.close()