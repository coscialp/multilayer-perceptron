import json
import sys, os
import matplotlib.pyplot as plt 
import numpy as np
from AI_learn.neural_network import MLPClassifier
from AI_learn.dataset import load_dataset


def verif_arg() -> bool:
    number_of_arg = len(sys.argv)
    save = False

    if number_of_arg < 3 or number_of_arg >= 5:
        print('Wrong number of arguments')
        exit(1)

    if number_of_arg == 4:
        if sys.argv[3] == '--save' or sys.argv[3] == '-s':
            save = True
        else:
            print(f'\'{sys.argv[3]}\' is not a valid option!')
            exit(1)

    if sys.argv[1] != 'fit' and sys.argv[1] != 'predict':
        print(f'\'{sys.argv[1]}\' is not a valid option!')
        exit(1)

    if sys.argv[2].split('.')[-1] != 'csv':
        print(f'\'{sys.argv[2]}\' is not a csv file!')
        exit(1)

    return save


def print_metrics(ax, fig, model, name_of_model, save):
    ax[0].plot(range(0, len(model.loss_) * 10, 10), model.loss_, label=f'loss_{name_of_model}')
    ax[1].plot(range(0, len(model.val_loss_) * 10, 10), model.val_loss_, label=f'val_loss_{name_of_model}')
    ax[2].plot(range(0, len(model.acc_) * 10, 10), model.acc_, label=f'acc_{name_of_model}')

    ax[0].set_xlabel('N iterations')
    ax[1].set_xlabel('N iterations')
    ax[2].set_xlabel('N iterations')

    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Validation loss')
    ax[2].set_ylabel('Accuracy')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    file = None
    if save == True:
        try:
            os.mkdir(f'{os.path.abspath(".")}/metrics')
        except FileExistsError:
            pass
        try:
            file = open(f'metrics/{name_of_model}.metrics', 'x')
        except FileExistsError:
            file = open(f'metrics/{name_of_model}.metrics', 'w')
        plt.savefig('metrics/figures.png')

    print(f'Metrics of {name_of_model}\'s model:')
    print('-' * 100)
    for epoch, loss in enumerate(model.loss_):
        line = f'epoch {epoch + 1}/{len(model.loss_)} - loss: {loss:.5f} - val_loss: {model.val_loss_[epoch]:.5f} - acc: {model.acc_[epoch]:.5f}'
        if save == True:
            file.write(f'{line}\n')
        print(line)
    print('-' * 100)


def save_model(best_model):
    with open('model.npy', 'wb') as file:
        np.save(file, best_model.hidden_layers_)
        np.save(file, len(best_model.parameters_) // 2)
        np.save(file, best_model.normalize_)
        if best_model.normalize_ == True:
            np.save(file, best_model.normalize_mean_)
            np.save(file, best_model.normalize_std_)
        for c in range(1, len(best_model.parameters_) // 2):
            np.save(file, best_model.parameters_[f'W{c}'])
            np.save(file, best_model.parameters_[f'b{c}'])


def load_model():
    hidden_layer = None
    mean = None
    std = None
    parameters = {}
    with open('model.npy', 'rb') as file:
        hidden_layer = np.load(file)
        C = np.load(file)
        normalize = np.load(file)
        if normalize == True:
            mean = np.load(file)
            std = np.load(file)
        for c in range(1, C):
            parameters[f'W{c}'] = np.load(file)
            parameters[f'b{c}'] = np.load(file)

    return (hidden_layer, mean, std, parameters)


if __name__ == '__main__':
    save = verif_arg()
        
    dataset = load_dataset(sys.argv[2], y_name='Diagnosis', indesirable_feature=['Index'])

    X = dataset.data.T
    y = dataset.target.T

    if sys.argv[1] == 'fit':
        adam = MLPClassifier(hidden_layers=(128, 128, 128), n_iter=2000, learning_rate=0.001, normalize=True, early_stopping=False)
        adam.fit(X, y, solver='adam', random_state=0)

        sgd = MLPClassifier(hidden_layers=(128, 128, 128), n_iter=2000, learning_rate=0.001, normalize=True, early_stopping=False)
        sgd.fit(X, y, solver='sgd', random_state=0)

        RMSprop = MLPClassifier(hidden_layers=(128, 128, 128), n_iter=2000, learning_rate=0.001, normalize=True, early_stopping=False)
        RMSprop.fit(X, y, solver='RMSprop', random_state=0)

        fig, ax = plt.subplots(1, 3, figsize=(14, 7))

        best_model = adam
        best_model_name = 'adam'
        
        if best_model.acc_[-1] < sgd.acc_[-1]:
            best_model = sgd
            best_model_name = 'sgd'
        if best_model.acc_[-1] < RMSprop.acc_[-1]:
            best_model = RMSprop
            best_model_name = 'RMSprop'

        save_model(best_model)

        print_metrics(ax, fig, adam, 'adam', save)
        print_metrics(ax, fig, sgd, 'sgd', save)
        print_metrics(ax, fig, RMSprop, 'RMSprop', save)

        print(f'The {best_model_name} model is the best. So we keep its parameters')

        plt.show()
    elif sys.argv[1] == 'predict':
        hidden_layer, mean, std, parameters = load_model()

        print(hidden_layer)
        print(mean)
        print(std)
        for key, val in parameters.items():
            print(f'{key}: {val.shape}')

        