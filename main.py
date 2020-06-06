import os
import argparse
import warnings

import numpy as np
from scipy.io import wavfile

from hmmlearn import hmm
from python_speech_features import mfcc


def build_arg_parser():
    """Функция для анализа аргументов"""
    parser = argparse.ArgumentParser(description='Trains the HMM-based speech recognition system')
    parser.add_argument('--input-folder', dest='input_folder', required=True, help='Input folder with audio files for training')
    return parser


def build_models(input_folder):
    """Создает модель для каждого слова"""
    # Переменная для хранения всех моделей для каждого слова
    speech_models = []

    # Анализ входного каталога
    for dirname in os.listdir(input_folder):
        # Получаем подпапку
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue

        # Извлечение метки
        label = subfolder[subfolder.rfind('\\') + 1:]

        # Переменная для хранения тренировочных данных
        X = np.array([])

        # Создаем список файлов для тренировки моделей.
        # Один из файлов в каждой папке оставляем для тестирования.
        training_files = [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]

        # Проходимся по тренировочным файлам и строим модели
        for filename in training_files:
            # Извлекаем путь к текущему файлу
            filepath = os.path.join(subfolder, filename)

            # Читаем аудиосигнал из файла
            sampling_freq, signal = wavfile.read(filepath)

            # Извлекаем MFCC-признаки
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features_mfcc = mfcc(signal, sampling_freq)

            # Присоединим точку данных к переменной X
            if len(X) == 0:
                X = features_mfcc
            else:
                X = np.append(X, features_mfcc, axis=0)

            # Создание HMM-модели
            model = ModelHMM()

            # Обучение модели, используя тренировочные данные
            model.train(X)

            # Сохранение модели для текущего слова
            speech_models.append((model, label))

            # Сброс переменной
            model = None

    return speech_models


def run_test(test_files):
    """Функция для тестирования входных данных"""
    # Классификация входных данных
    for test_file in test_files:
        # Чтение входного файла
        sampling_freq, signal = wavfile.read(test_file)

        # Извлечение MFCC признаков
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(signal, sampling_freq)

        # Переменные для хранения максимальной оценки и выходной метки
        max_score = -float('inf')
        output_label = None

        # Выполняем итерации по моделям, чтобы выбрать наилучшую
        # Прогоняем текущий вектор призанков через каждую HMM-модель
        # выбирая ту из них, которая получит наивысшую оценку
        for item in speech_models:
            model, label = item

            # Вычислим оценку и сравним ее с максимальной оценкой
            score = model.compute_score(features_mfcc)
            if score > max_score:
                max_score = score
                predicted_label = label

        # Вывод предсказанного результата
        start_index = test_file.find('\\') + 1
        end_index = test_file.rfind('\\')+1
        print(test_file, start_index, end_index)
        original_label = test_file[end_index:]
        print('Original: {}'.format(original_label))
        print('Predicted: {}'.format(predicted_label))
        print()


class ModelHMM(object):
    """Класс для тренировки HMM"""

    def __init__(self, num_components=4, num_iter=1000):
        self.n_components = num_components
        self.n_iter = num_iter

        # Определяем тип ковариации и тип HMM
        self.cov_type = 'diag'
        self.model_name = 'GaussianHMM'

        # Инициализируем переменную, в которой будут храниться модели для каждого слова
        self.models = []

        # Определяем модель, используя вышеуказанные параметры
        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter)

    def train(self, training_data):
        """
        Метод для обучения модели

        training_data - 2D массив numpy, каждая строка в нем имеет 13 измерений
        :param training_data:
        :return:
        """
        np.seterr(all='ignore')
        cur_model = self.model.fit(training_data)
        self.models.append(cur_model)

    def compute_score(self, input_data):
        """
        Выполнение HMM модели для оценки входных данных

        :param input_data:
        :return:
        """
        return self.model.score(input_data)


# Определим основную функцию и получим входную папку из входного параметра
if __name__ == '__main__':
    input_folder = 'data'
    # args = build_arg_parser().parse_args()
    # input_folder = args.input_folder

    # Создаем HMM-модель для каждого слова из входной папки
    speech_models = build_models(input_folder)

    # Один из файлов в каждой папке остается для тестирования. Используем его, что выяснить
    # насколько точна данная модель. Тестовые файлы в нашем случае - 15-й файл в каждой папке
    test_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if '15' in x):
            filepath = os.path.join(root, filename)
            test_files.append(filepath)

    run_test(test_files)
