import sys, os
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
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
        print_metrics(ax, fig, adam, 'adam', save)
        print_metrics(ax, fig, sgd, 'sgd', save)
        print_metrics(ax, fig, RMSprop, 'RMSprop', save)
        plt.show()
    elif sys.argv[1] == 'predict':
        pass