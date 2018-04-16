import numpy as np


def apply_along_axis_matrix(func, matrix_1, matrix_2, axis=None, *args, **kwargs):
    """
    :param func: this function should accept two 1xD arrays.
     It is applied to 1xD slices of matrix_1, matrix_2 along the specified axis.
    :param matrix_1: matrix NxD
    :param matrix_2: matrix MxD
    :param axis: axis along which matrix_1 and matrix_2 are sliced.
    :return: matrix NxM of results from func
    """

    def apply(m_2, m_1):
        return func(m_1, m_2, *args, **kwargs)

    def slice_second_matrix(vector, matrix):
        return np.apply_along_axis(apply, axis=axis, arr=matrix, m_1=vector)

    return np.apply_along_axis(slice_second_matrix, axis=axis, arr=matrix_1, matrix=matrix_2)


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """

    y = p_y_x.shape[1] - np.argmax(np.flip(p_y_x, axis=1), axis=1) - 1
    bool_y = np.not_equal(y, y_true)
    return np.mean(bool_y)


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """

    length = np.max(ytrain)+1
    env = np.bincount(ytrain, minlength=length)

    return env/len(ytrain)


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) (czy p(x=1|y)?) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y (czy p_x_1_y?)  o wymiarach MxD.
    """

    length = np.max(ytrain) + 1
    denominator = np.bincount(ytrain, minlength=length) + a + b - 2

    def quotient(x):
        numerator = np.bincount(np.extract(x, ytrain), minlength=length) + a - 1
        # TODO: cos tam cos tam
        return numerator/denominator

    return np.apply_along_axis(quotient, axis=0, arr=Xtrain.toarray())


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """

    def beta(x, theta):
        theta += 1e-10
        return np.prod((theta ** x) * (1 - theta) ** (1 - x))

    def divide(x):
        x_sum = np.sum(x)
        if x_sum != 0:
            return x/x_sum
        return x*0

    p_y_x_p_x = apply_along_axis_matrix(beta, X.toarray(), p_x_1_y, axis=1) * p_y
    p_y_x = np.apply_along_axis(divide, axis=1, arr=p_y_x_p_x)

    return p_y_x


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """

    results = []
    errors = []
    for a in a_values:
        error = []
        for b in b_values:
            p_y = estimate_a_priori_nb(ytrain)
            p_x_1_y = estimate_p_x_y_nb(Xtrain, ytrain, a, b)

            p_y_x = p_y_x_nb(p_y, p_x_1_y, Xval)
            err = classification_error(p_y_x, yval)

            error.append(err)
            results.append((err, a, b))
        errors.append(error)

    error_best, best_a, best_b = min(results, key=lambda x: x[0])

    return error_best, best_a, best_b, errors
