# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(252, 503):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(GRU(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(GRU(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(GRU(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(GRU(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

# # Fitting the RNN to the Training set
epochs = 10
batch_size = 32
regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
##############################################################
################### Loop de treinamento ######################
##############################################################

# retirar o comentário para carregar rede previamente treinada
# m.load(save_file)


# Define o número de vezes que a rede neural observara todas
# as Sequências
# for _ in range(epochs):
#
# 	# Treina em mini-lotes e utiliza todos os dados
# 	try:
# 		regressor.fit(X_train, y_train, validation_set=0.01, batch_size=batch_size,
# 			  n_epoch=1, snapshot_epoch=True, run_id='LiterNet')
#
# 	except KeyboardInterrupt:
# 		# aborta com ctrl+c
# 		break

	# m.save(save_file) # salva o modelo

	# Cria uma sequência de caracteres a partir do texto
	# A RNR utilizará essa sequência como ponto de partida
	# seed = random_sequence_from_textfile(path, maxlen)

	# Gera texto. Começaremos apresentando a sequência
	# feita acima. A seguir, a rede nós dará o próximo
	# caractere. Nós então colocaremos esse caractere no
	# final da sequência de inicialização e continuaremos
	# prevendo mais e mais caracteres.

	# print ("\n\n\n-- Testando...")
	# print ("\n-- Teste com temperatura de 1.0 --\n")
	# print (m.generate(1000, temperature=1.0, seq_seed=seed))
    #
	# print ("\n-- Teste com temperatura de 0.5 --\n")
	# print (m.generate(1000, temperature=0.5, seq_seed=seed))
    #
	# print ("\n-- Teste com temperatura de 0.25 --\n")
	# print (m.generate(1000, temperature=0.25, seq_seed=seed))
