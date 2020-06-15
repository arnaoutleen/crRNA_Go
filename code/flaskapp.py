import pandas as pd
import cv2
from PIL import Image
import numpy as np
import csv
import cgi, os
from Bio.Seq import Seq
from sklearn.model_selection import train_test_split, ShuffleSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, MaxPooling1D, Conv1D, Flatten, Bidirectional
import matplotlib.pyplot as plt
from numpy import argmax
from Bio.Graphics import GenomeDiagram
from reportlab.lib import colors
from reportlab.lib.units import cm
from Bio import SeqIO, SeqFeature
from Bio.SeqFeature import SeqFeature, FeatureLocation
from flask import Flask, flash, request, redirect, url_for, render_template
from flask_wtf import Form
from werkzeug.utils import secure_filename
from time import time

#declare app name, and upload folder
#change this upload folder directory to where you are hosting your uploaded files
#ideally it should be your webapp's folder
app = Flask(__name__)
UPLOAD_FOLDER = '/home/cornelia/Desktop/flaskapp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

####################################################################################

#declare functions that you'll use later on
#function that converts DNA input to RNA seqn + all uppercase
def DNA_to_RNA(input_seqn):
	old_seqn=str(input_seqn)
	old_seqn = old_seqn.strip('[')
	old_seqn = old_seqn.strip(']')
	old_seqn = old_seqn.strip("'")
	old_seqn = old_seqn.replace("\r","")
	old_seqn = old_seqn.replace("\n","")
	old_seqn = old_seqn.replace(" ","")
	old_seqn = old_seqn.replace("\\","")
	old_seqn = old_seqn.replace("n","")
	data=old_seqn.replace('T','U').upper()
	return(data)

#function that one-hot encodes the data
def one_hot_this(data):
	alphabet = 'AUGC'
	onehot_encoded_1=[]
	seqn_array_1=[]
	# define a mapping of chars to integers
	char_to_int_1 = dict((c, i) for i, c in enumerate(alphabet))
	int_to_char_1 = dict((i, c) for i, c in enumerate(alphabet))
	# integer encode input data
	integer_encoded_1 = [char_to_int_1[char] for char in data]
	# print(integer_encoded)
	# one hot encode
	onehot_encoded_1 = list()
	for value in integer_encoded_1:
	    letter_1 = [0 for _ in range(len(alphabet))]
	    letter_1[value] = 1
	    onehot_encoded_1.append(letter_1)
	return(onehot_encoded_1)

####################################################################################

#declare pages, routes and functions associated w routes

@app.route("/uploads")
def upload():
	return('uploads page!')

#home page defaults to training page
@app.route("/")
@app.route("/train")
def train():
	return render_template('train.html')

#training page and home page after you submit query, POST method
@app.route("/", methods=['GET', 'POST'])
@app.route("/train", methods=['GET', 'POST'])
def train_2():
	if request.method=='POST':
		#from the training website, you POST:
		# the csv training set, model name, number of epochs and batches,
		#and train the model using these values
		if request.files: 
			file = request.files["setcsv"]
			filename = secure_filename(file.filename)
			basedir = os.path.abspath(os.path.dirname(__file__))
			current_time = str(int(time()))
			a = file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
		#read input csv file for training
		data=pd.read_csv(filename, sep=',')

		#split .csv file into scores and DNA sequences
		energetics_scores=data[['Access 8nts','Access 16nts','Energy A.','Sequence A.','Self Folding','Free End']]
		sequences=data[['target seqn']]

		#sum up energetics scores in each row to one score
		for row in energetics_scores:
			rowtotal=0
			for column in row[1:]:
				rowtotal = (energetics_scores.sum(axis=1))
				rowtotal = rowtotal / 60000

		#print the sum to one numpy array for later use
		refined_with_sum=energetics_scores.sum(axis=1).to_numpy()
		refined_with_sum=refined_with_sum / 60000
		#convert each DNA sequence into RNA + one hot encode  
		sequences=sequences.values.tolist()
		length=len(sequences)
		seqn_array=[]
		for i in range(length):
			#convert to RNA sequence by replacing T with U
			data = DNA_to_RNA(sequences[i])
			one_hot_encoded = one_hot_this(data)
			seqn_array.append(one_hot_encoded)

		# #################################################################################################
		# now, insert ML algo


		# #split into train and test groups
		# #convert np array to tensor
		x = np.asarray(seqn_array)
		y = np.asarray(refined_with_sum)
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
		x_train=tf.convert_to_tensor(x_train,dtype=tf.int32)
		y_train=tf.convert_to_tensor(y_train,dtype=tf.float32)
		x_test=tf.convert_to_tensor(x_test,dtype=tf.int32)
		y_test=tf.convert_to_tensor(y_test,dtype=tf.float32)

		#build model
		model=Sequential()
		model.add(Bidirectional(LSTM(20, activation='relu', return_sequences=True),input_shape=x_train.shape[1:]))
		model.add(Dropout(0.2))
		model.add((LSTM(10, activation='relu')))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(1, activation='relu'))
		# compile the layer
		opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
		# mean squared error = mse
		model.compile(loss='binary_crossentropy',
			optimizer=opt,
			metrics=['accuracy'])
		epochs = int(request.form['epochs'])
		batch = int(request.form['batch'])
		model_name = str(request.form['modelname'])
		history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),verbose=1,batch_size=batch)
		current_time = str(int(time()))
		model.save("./static/" + current_time + model_name + ".h5")
		#######################################################################################################

		# # save model output accuracy vals as images
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		current_time = str(int(time()))
		plt.savefig("./static/" + current_time + "loss.png")
	#save the loss graph and training model with a timestamp to static file, and output those to returned page
	return render_template('train_results.html', lossgraph=str("/static/" + current_time + "loss.png"), downloadmodel=str("/static/" + current_time + model_name + ".h5"))

# sample template for training results and binding site input page
@app.route("/train/results")
def train_results():
	return render_template('train_results.html')

@app.route("/binding")
def binding():
	return render_template('binding site finder.html')

# finding the binding site using POST method
@app.route("/binding",methods=['GET', 'POST'])
def binding_2():
	# you post the model you're using to predict, the seqn you're predicting for, and length of bind site 
	if request.method=='POST':
		if request.files: 
			file = request.files["file"]
			filename = secure_filename(file.filename)
			basedir = os.path.abspath(os.path.dirname(__file__))
			current_time = str(int(time()))
			a = file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
			
		model = load_model(filename)

		## read DNA sequence in x bp increment, write to csv, with location and direction
		# load sequence
		old_seqn=Seq(request.form['seqn_input'])
		old_rc=old_seqn.reverse_complement()
		old_seqn_str=str(old_seqn)
		old_rc_str=str(old_rc)
		#convert to RNA sequence by replacing T with U
		new_seqn_str=old_seqn_str.replace('T','U').upper()
		new_rc_str=old_rc_str.replace('T','U').upper()
		length_RNA=len(new_rc_str)
		steps = int(request.form['size'])

		#set up loop vars for fwd loop indexing
		start_index=0
		end_index= steps
		temp=[]
		temp_2=[]
		fwd_strings=[]
		fwd_strand=[]
		fwd_position=[]
		rev_strings=[]
		rev_strand=[]
		rev_position=[]

		#fwd RNA segmentation + writing to array
		for i in range(length_RNA):
			if end_index > length_RNA:
				break
			temp=new_seqn_str[start_index:end_index]
			#print(temp)
			fwd_strand.append('+')
			fwd_strings.append(temp)
			fwd_position.append(start_index + 1)
			start_index = start_index + 1
			end_index = end_index + 1

		df=pd.DataFrame({'Sequence' : fwd_strings, 'Strand' : fwd_strand,'Position' : fwd_position})
		
		#reset loop vars
		start_index=0
		end_index= steps
		rev_index=length_RNA

		#rev RNA segmentation + writing to array
		for i in range(length_RNA):
			if end_index > length_RNA:
				break
			temp_2=new_rc_str[start_index:end_index]
			#print(temp_2)
			rev_strings.append(temp_2)
			rev_strand.append('-')
			rev_index = length_RNA - end_index
			rev_position.append(rev_index + 1)
			start_index = start_index + 1
			end_index = end_index + 1

		df_2=pd.DataFrame({'Sequence' : rev_strings, 'Strand' : rev_strand,'Position' : rev_position})
		new_df=df.append(df_2,ignore_index=True)
		
		# # one-hot encode all outputs into a matrix
		RNA_inputs=new_df[['Sequence']]

		#convert each RNA sequence into AUGC + one hot encode

		RNA_inputs=RNA_inputs.values.tolist()
		length=len(RNA_inputs)
		seqn_array=[]

		for i in range(length):
			data = DNA_to_RNA(RNA_inputs[i])
			one_hot_encoded = one_hot_this(data)
			seqn_array.append(one_hot_encoded)
			
		seqn_array = np.asarray(seqn_array)	
		seqn_array = np.expand_dims(seqn_array, axis=0)
		#input into prediction module
		predict_energy=[]
		size_of_arg=len(seqn_array)

		for k in range(size_of_arg):
			temp_energy = model.predict(seqn_array[k])
			predict_energy.append(temp_energy)
		predict_energy = np.asarray(predict_energy)
		predict_energy = predict_energy[0, :, :]
		# write to dataframe
		new_df['Predicted Energy Score'] = predict_energy 
		# order by descending energy score
		sorted_results = new_df.sort_values('Predicted Energy Score',ascending=False)
		sorted_results = sorted_results.reset_index(drop=True)
		
		# write to csv file
		current_time = str(int(time()))
		download_file = sorted_results.to_csv("./static/" + current_time + ".csv", index=False)
		
		# make a diagram for the top five hits 
		gdd = GenomeDiagram.Diagram('Test Diagram')
		number = 5
		color=[colors.blue, colors.red, colors.green, colors.orange, colors.purple]
		# steps
		for i in range(number):
			gdt_features_i = gdd.new_track(i+1, greytrack=False)
			gds_features_i = gdt_features_i.new_set()
			temp_start = sorted_results.loc[i+1,"Position"]
			if sorted_results.loc[i+1,"Strand"] == "+":
				temp_end = temp_start + steps
				strand_val = +1
				strand_name = "crRNA{}".format(5-i)
			else:
				temp_end = temp_start + steps
				strand_val = -1
				strand_name = "crRNA{}".format(5-i)
			feature = SeqFeature(FeatureLocation(int(temp_start), int(temp_end), strand=strand_val))
			gds_features_i.add_feature(feature, name=strand_name, label=True, label_size=25, label_angle = 0, sigil="ARROW",color=color[i])	
		#build and save image
		gdd.draw(format='linear', pagesize='A3', fragments=1,
		         start=0, end=length_RNA)
		current_time = str(int(time()))
		image = gdd.write("./static/" + current_time + ".jpg", "jpg")
		basedir_1 = os.path.abspath(os.path.dirname(__file__))
	#save image and csv of results
	return render_template('binding site results.html', download_file=str("/static/" + str(int(time())) + ".csv"), image=str("/static/" + current_time + ".jpg")) # os.path.join(basedir, app.config['UPLOAD_FOLDER'], 

# sample RNA sequence to copy and paste - comes from SARS-nCOV-2
@app.route("/binding/sampleRNA")
def copy():
	return ("Copy and paste this next sequence into the text box: \n \n  AUUAAAGGUUUAUACCUUCCCAGGUAACAAACCAACCAACUUUCGAUCUCUUGUAGAUCUGUUCUCUAAACGAACUUUAAAAUCUGUGUGGCUGUCACUCGGCUGCAUGCUUAGUGCACUCACGCAGUAUAAUUAAUAACUAAUUACUGUCGUUGACAGGACACGAGUAACUCGUCUAUCUUCUGCAGGCUGCUUACGGUUUCGUCCGUGUUGCAGCCGAUCAUCAGCACAUCUAGGUUUCGUCCGGGUGUGACCGAAAGGUAAGAUGGAGAGCCUUGUCCCUGGUUUCAACGAGAAAACACACGUCCAACUCAGUUUGCCUGUUUUACAGGUUCGCGACGUGCUCGUACGUGGCUUUGGAGACUCCGUGGAGGAGGUCUUAUCAGAGGCACGUCAACAUCUUAAAGAUGGCACUUGUGGCUUAGUAGAAGUUGAAAAAGGCGUUUUGCCUCAACUUGAACAGCCCUAUGUGUUCAUCAAACGUUCGGAUGCUCGAACUGCACCUCAUGGUCAUGUUAUGGUUGAGCUGGUAGCAGAACUCGAAGGCAUUCAGUACGGUCGUAGUGGUGAGACACUUGGUGUCCUUGUCCCUCAUGUGGGCGAAAUACCAGUGGCUUACCGCAAGGUUCUUCUUCGUAAGAACGGUAAUAAAGGAGCUGGUGGCCAUAGUUACGGCGCCGAUCUAAAGUCAUUUGACUUAGGCGACGAGCUUGGCACUGAUCCUUAUGAAGAUUUUCAAGAAAACUGGAACACUAAACAUAGCAGUGGUGUUACCCGUGAACUCAUGCGUGAGCUUAACGGAGGGGCAUACACUCGCUAUGUCGAUAACAACUUCUGUGGCCCUGAUGGCUACCCUCUUGAGUGCAUUAAAGACCUUCUAGCACGUGCUGGUAAAGCUUCAUGCACUUUGUCCGAACAACUGGACUUUAUUGACACUAAGAGGGGUGUAUACUGCUGCCGUGAACAUGAGCAUGAAA")

#sample empty binding results page
@app.route("/binding/results")
def binding_results():
	return render_template('binding site results.html')

#empty crRNA designer page
@app.route("/designer")
def designer():
	output = ""
	return render_template('designer.html', output = output)

#crRNA designer sample RNA to use to test functionality
@app.route("/designer/sampleRNA")
def sample():
	return ("Copy and paste this next sequence into the text box: \n \n  CGTATGGTCCACTGCTGATTTTA")

#crRNA designer page after you post your input sequence 
@app.route("/designer",methods=['GET', 'POST'])
def designer2():
	if request.method == 'POST':
		old_seq=request.form['input']
		new_seq=Seq(old_seq)
		new_seq=str(new_seq.reverse_complement())
		new_seq=new_seq.replace('T','U').upper()
		output = "Here is your resulting sequence: 3'-ACAACAUUAUCGGGGGUUUUGACCUGGAAGGUGUUG"+new_seq+"-5'"
		#this script implements your input seqn into the chassis of the crRNA
	return render_template('designer.html', output = output)

### turn on debugger mode if running directly on python
if '__name__' == '__main__':
	app.run(debug=True)