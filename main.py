from scipy import sparse
from data_extraction import *
from machine_learning_algorithms import *
import pickle
from html_extraction import *


test_path = r'test\1.txt'
train_directory = r'train'
val_directory = r'val'


def learn():
    x_train, y_train, labels = create_data(train_directory)
    x_val, y_val, _ = create_data(val_directory)

    x_train = sparse.csr_matrix(x_train)
    x_val = sparse.csr_matrix(x_val)

    a_values = [1, 3, 10, 30, 100, 300, 1000]
    b_values = [1, 3, 10, 30, 100, 300, 1000]

    _, a, b, _ = model_selection_nb(x_train, x_val, y_train, y_val, a_values, b_values)

    priori = estimate_a_priori_nb(y_train)
    posteriori = estimate_p_x_y_nb(x_train, y_train, a, b)

    return priori, posteriori, labels


def best_label(x_test, priori, posteriori):
    x_test = sparse.csr_matrix(x_test)
    probabilities = p_y_x_nb(priori, posteriori, x_test)
    return np.argmax(probabilities)


def save_data(data):
    with open('data.pkl', 'wb') as file:
        pickle.dump(data, file)


def load_data():
    with open('data.pkl', 'rb') as file:
        return pickle.load(file)


def save_model():
    priori, posteriori, labels = learn()
    save_data((priori, posteriori, labels))


def main():
    priori, posteriori, labels = load_data()

    text = read_file(test_path).splitlines()
    x_test = extract(text, get_wordlist(wordlist_path))
    index_of_label = best_label(x_test, priori, posteriori)

    jobs = job_list(labels[index_of_label], 'wroclaw')
    save_links(jobs)


def save_links(data):
    with open('links.txt', 'wb') as file:
        for name, link in data:
            file.write('{};;{}\n'.format(name, link).encode())


if __name__ == '__main__':
    # save_model()
    main()
