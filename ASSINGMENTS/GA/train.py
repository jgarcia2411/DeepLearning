#-----------------------Libraries----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from random import random, shuffle,sample, randrange, uniform
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input, Dense, Dropout

from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split

#-----------------------Data----------------------------------------
model_data = pd.read_csv('model_data.csv', index_col=0)
# Create features and label dataset. NOTE: We're predicting a value 24 hours into the future.
features = model_data # All of the above columns and remove readings from last 24 hours
target = model_data.iloc[:, 0] # Appliances or the target/label column and remove readings from first 24 hours

#-----------------------LSTM input----------------------------------------
# split into train and test sets
trainX, testX, trainY, testY = train_test_split(features, target, test_size=0.30, random_state=42, shuffle = False)
windowlength = 336 #2 weeks
batch_size = 32
train_generator = TimeseriesGenerator(trainX, trainY, length=windowlength, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(testX, testY, length=windowlength, sampling_rate=1, batch_size=batch_size)
train_X, train_y = train_generator[0]
test_X, test_y = test_generator[0]

#train_samples = train_X.shape[0]*len(train_generator)
#test_samples = test_X.shape[0]*len(test_generator)

#print("Total Records (n): {}".format(model_data.shape))
#print("Total Records after adjusting for 24 hours: {}".format(len(features)))
#print("Number of samples in training set (.8 * n): trainX = {}".format(trainX.shape[0]))
#print("Number of samples in testing set (.2 * n): testX = {}".format(testX.shape[0]))
#print("Size of individual batches: {}".format(test_X.shape[1]))
#print("Number of total samples in training feature set: {}".format(train_samples))
#print("Number of samples in testing feature set: {}".format(test_samples))

#-----------------------Base model----------------------------------------
# design network
#net_units = 15
#net_num_epoch = 5000
#net_learning_rate = 0.0013779778182843216
#0.00144

#model = Sequential()
#model.add(LSTM(net_units, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(LeakyReLU(alpha=0.5))
#model.add(Dropout(0.40))
#model.add(Dense(1))

#adam = Adam(lr= net_learning_rate)
# Stop training when a loss function is not improving
#callback = [EarlyStopping(monitor='val_loss', min_delta = 0.00001, patience= 20, mode = 'auto', restore_best_weights=True)]
#model.compile(loss='mae', optimizer='adam', metrics = ['mae'])
# fit network
#history = model.fit_generator(
#    train_generator,
#    epochs=net_num_epoch,
#    validation_data=test_generator,
#    callbacks = callback,
#    verbose=2,
#    shuffle=False,
#    initial_epoch=0
#    )
#score = model.evaluate_generator(test_generator, verbose=0)
#print(f'Test Loss = {score[0]}, MAE = {score[1]}')
# plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()

#-----------------------Predictions----------------------------------------

#yhat_train_temp = model.predict_generator(train_generator)
#yhat_test_temp = model.predict_generator(test_generator)

#n_lead = 1

#yhat_train = yhat_train_temp[:, n_lead - 1]
#yhat_test = yhat_test_temp[:, n_lead - 1]

# Shift predictions for plotting

# training results
#yhat_train_plot = np.empty(shape=[target.shape[0], ])
#yhat_train_plot[:] = np.nan
#yhat_train.shape = yhat_train.shape[0]
#yhat_train_plot.shape = yhat_train_plot.shape[0]
#yhat_train_plot[windowlength:len(yhat_train) + windowlength] = yhat_train

# test results
#yhat_test_plot = np.empty(shape=[target.shape[0], ])
#yhat_test_plot[:] = np.nan
#yhat_test.shape = yhat_test.shape[0]
#yhat_test_plot.shape = yhat_test_plot.shape[0]
#yhat_test_plot[len(yhat_train) + (windowlength * 2):len(target)] = yhat_test

#fig = plt.figure()
#plt.style.use('seaborn')
#palette = plt.get_cmap('Set1')
#pyplot.plot(y[:, n_lead-1], marker='', color=palette(4), linewidth=1, alpha=0.9, label='actual')
#plt.plot(target, marker='', color=palette(4), linewidth=1, alpha=0.9, label='actual')
#plt.plot(yhat_train_plot, marker='', color=palette(2), linewidth=1, alpha=0.9, label='training predictions')
#plt.plot(yhat_test_plot, marker='', color=palette(3), linewidth=1, alpha=0.9, label='testing predictions')

#plt.title('Appliances Energy Prediction', loc='center', fontsize=20, fontweight=5, color='orange')
#plt.ylabel('Energy used (Wh)')
#plt.legend()
#fig.set_size_inches(w=20,h=10)
#plt.show()

#-----------------------Network architecture optimization----------------------------------------

class lstmGA():

    def __init__(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            population_size=10,
            num_iter=50,
            keep_top_n=0.5,
            mutation_rate=0.5):

        self.population_size = population_size
        self.num_iter = num_iter
        self.keep_top_n = keep_top_n
        self.mutation_rate = mutation_rate
        self._iteration_progress = []
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # create a initial starting popultaion
        self.starting_states = self._generate_start_state()

    def _generate_start_state(self):
        final_population = []
        for _ in range(self.population_size):
            epochs = randrange(0, 500, 10)
            batch_size = randrange(10, 500, 20)
            n_units = randrange(0, 200, 10)
            learning_rate = uniform(0.00001, 0.2)
            dropout = uniform(0, 0.6)
            final_population.append([epochs, batch_size, n_units, learning_rate, dropout])
        return final_population

    def _evaluate(self, individual):

        #train_generator = TimeseriesGenerator(self.x_train, self.y_train, length=individual[1], sampling_rate=1,
                                             # batch_size=individual[1])

        #test_generator = TimeseriesGenerator(self.x_test, self.y_test, length=individual[1], sampling_rate=1,
                                             #batch_size=individual[1])
        #train_X, train_y = train_generator[0]
        #test_X, test_y = test_generator[0]

        model = Sequential()
        model.add(LSTM(
            int(individual[2]),
            input_shape=(train_X.shape[1], train_X.shape[2])
        ))
        model.add(LeakyReLU(alpha=0.5))
        model.add(Dropout(individual[-1]))
        model.add(Dense(1))

        adam = Adam(learning_rate=individual[3])
        # Stop training when a loss function is not improving
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        model.fit(
            train_generator,
            epochs=int(individual[0]),
            validation_data=test_generator,
            verbose=0,  # 2
            shuffle=False)

        results = model.evaluate(
            test_generator,
            verbose=0,  # 1
            return_dict=True
        )
        return results

    def evaluate(self, final_population):
        fitness_results = []
        for individual in final_population:
            metrics = self._evaluate(individual)
            fitness_results.append(metrics['mse'])
        print('State: evaluating')
        return fitness_results

    def reduce_population(self, population, fitness_results):
        # Sort the results and get the indices
        # in order to get the right population rows
        sorted_indices = np.argsort(fitness_results)
        # Determine how many top population members
        # to keep based on if 'keep_top_n' is a float
        # or an integer
        if isinstance(self.keep_top_n, float):
            top_n = max(int(self.keep_top_n * len(fitness_results)), 1)
        else:
            top_n = self.keep_top_n
        final_indices = sorted_indices[:top_n]

        # Pull the appropiate population rows
        reduced_population = np.array(population)[final_indices].tolist()
        print('State: Reducing population')
        return reduced_population

    def crossover(self, reduced_population, r_cross=0.6):
        # This function crossover two parents to create two children
        parents = sample(reduced_population, 2)  # list of tuples, each tuple represent two parents
        # parents will have 2 children and will be stored in offpring:
        c1 = parents[0]
        c2 = parents[1]

        if np.random.rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(1, len(parents[0]) - 2)
            # perform crossover
            c1 = parents[0][:pt] + parents[1][pt:]
            c2 = parents[1][:pt] + parents[0][pt:]
        offspring_list = [c1, c2]
        print('State: Crossover')
        return offspring_list

    def _mutate(self, offspring_list):
        for individual in offspring_list:
            for i in range(len(individual)):
                # check for a mutation
                if np.random.rand() < self.mutation_rate:
                    # mutate randomly
                    if i < 3:
                        individual[i] = int(random.choice([0.5, 1, 1.5]) * individual[i])
                    else:
                        individual[i] = random.choice([0.5, 1, 1.5]) * individual[i]
        return offspring_list

    def procreate(self, reduced_population):
        reduced_population_size = len(reduced_population)
        new_population = []

        while len(new_population) < self.population_size - reduced_population_size:
            new_population.extend(self.crossover(reduced_population))
        new_population = self._mutate(new_population)
        print('State:procreating')
        return new_population

    def optimize(self, verbose=False):
        'Run the optimization algorithm'
        best_mse = 1E10
        best_sequence = []

        # Optimize using genetic algorithm
        for i in range(self.num_iter):
            if verbose:
                print(f'OPTIMIZING ROUND {i + 1}/{self.num_iter}'.center(100, '='))
            # We want to make sure to keep population size the same
            # so we only pull the las 'population_size' rows.
            if verbose:
                print('Procreating...')
            starting_idx = -self.population_size
            population = (
                self.starting_states if i == 0  # randomly generates a starting population.
                else self.procreate(reduced_population)  # [starting_idx:]
            )
            # Evaluate sequences
            if verbose:
                print('Evaluating...')
            fitness_results = self.evaluate(
                population)  # list of the mse results from LSTM algorithm (fitness function)

            # Sort the results and get best resulting index
            best_idx = np.argsort(fitness_results)[0]  # select the best individual

            # Get best mse and sequence
            iter_best = fitness_results[best_idx]
            is_better = iter_best < best_mse  # compare with the last best result and keep it if the mse is lower than the previous.
            best_mse = iter_best if is_better else best_mse  # take the new best mse value
            best_sequence = self.starting_states[
                best_idx] if is_better else best_sequence  # select the best individual from starting_states

            # Reduce the population to best
            if verbose:
                print('Selecting best...')
            reduced_population = self.reduce_population(population, fitness_results)

            # Add iteration results to internal results tracker
            self._iteration_progress.append(best_mse)

            if verbose:
                # print best result of this iteration
                print(f'Best mse achieved: {best_mse} - {best_sequence}')
                print('=' * 100)
        print('State: optimizing')
        print(f'Best Overall:{best_mse} - {best_sequence}')

        # Generate graph of optimization progress.
        fig, ax = plt.subplots()
        sns.lineplot(
            x=range(1, len(self._iteration_progress) + 1),
            y=self._iteration_progress,
            ax=ax
        );
        ax.set_title('MSE vs Iteration Number')


GA = lstmGA(
    trainX,
    trainY,
    testX,
    testY,
    population_size=20,
    num_iter=5,
    keep_top_n=0.5,
    mutation_rate=0.5
)
GA.optimize(verbose=True)