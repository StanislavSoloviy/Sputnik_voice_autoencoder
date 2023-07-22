import csv
import numpy as np
import pandas as pd
from random import shuffle
import scipy
import librosa
import librosa.display
from sklearn import preprocessing
import wandb

class Model:
    def __init__(self):
        self.__data = None


    def prepare_database(self):
        """Метод для подготовки базы данных путём сопоставления 2х аудиовайлов и создания метки"""

        with open('prepearing.csv', encoding='cp1251') as file:
            rows = csv.DictReader(file, delimiter=';', quotechar='"')
            dict_data = {row['file']: row['index'] for row in rows}

        positive_list = [(key1, key2, True)   for key1, value1 in dict_data.items()
                                        for key2, value2 in dict_data.items()
                                        if value1 == value2]


        negative_list = [(key1, key2, False)   for key1, value1 in dict_data.items()
                                        for key2, value2 in dict_data.items()
                                        if value1 != value2]

        negative_list = negative_list[:len(positive_list)]
        data = positive_list + negative_list
        shuffle(data)
        self.__data = data



    @staticmethod
    def extract_features(files, directory, lbl):
        """Метод для извлечения значимых характеристик"""
        features = []
        for file in files:
            name = f'{directory}/{file}'
            y, sr = librosa.load(name, mono=True, duration=5)
            features.append(file)  # filename
            features.extend([np.mean(e) for e in librosa.feature.mfcc(y=y, sr=sr,
                                                                      n_mfcc=20)])  # mfcc_mean<0..20>
            features.extend([np.std(e) for e in librosa.feature.mfcc(y=y, sr=sr,
                                                                     n_mfcc=20)])  # mfcc_std
            features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                                    axis=0)[0])  # cent_mean
            features.append(np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                                   axis=0)[0])  # cent_std
            features.append(scipy.stats.skew(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                                             axis=0)[0])  # cent_skew
            features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                                    axis=0)[0])  # rolloff_mean
            features.append(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                                   axis=0)[0])  # rolloff_std

        features.append(lbl)
        return features

    def create_database(self):
        """Создание csv файла со значимыми характеристиками. 2аудиофайла идут подряд друг за другом в одной строке,
        в качестве метки - значение соответствия"""
        buffer = []
        buffer_size = 5000
        buffer_counter = 0

        # Создание заголовка для файла CSV.
        directory = "dataset"
        header = ['first_filename1']
        header.extend([f'first_mfcc_mean{i}' for i in range(1, 21)])
        header.extend([f'first_mfcc_std{i}' for i in range(1, 21)])
        header.extend(['first_cent_mean', 'first_cent_std', 'first_cent_skew', 'first_rolloff_mean', 'first_rolloff_std'])
        header.extend(['second_filename2'])
        header.extend([f'second_mfcc_mean{i}' for i in range(1, 21)])
        header.extend([f'second_mfcc_std{i}' for i in range(1, 21)])
        header.extend(['second_cent_mean', 'second_cent_std', 'second_cent_skew', 'second_rolloff_mean', 'second_rolloff_std'])
        header.extend(['label'])
        self.prepare_database()

        with open('dataset.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(header)
            for *files, lbl in self.__data:
                features = self.extract_features(files, directory=directory, lbl=lbl)
                if buffer_counter + 1 == buffer_size:
                    buffer.append(features)
                    writer.writerows(buffer)
                    print(f"- [{directory.split('/')[-1]}] Write {len(buffer)} rows")
                    buffer = []
                    buffer_counter = 0
                else:
                    buffer.append(features)
                    buffer_counter += 1
            if buffer:
                writer.writerows(buffer)
                print(f"- [{directory.split('/')[-1]}] Write {len(buffer)} rows")
            print(f"- [{directory.split('/')[-1]}] Writing complete")



    def create_and_train_model(self):
        """Метод создания модели методом опорных векторов"""
        data = pd.read_csv('dataset.csv', encoding='cp1251')
        y = data['label'].values

        X = data[['first_mfcc_mean1', 'first_mfcc_mean2', 'first_mfcc_mean3', 'first_mfcc_mean4',
                'first_mfcc_mean5', 'first_mfcc_mean6', 'first_mfcc_mean7', 'first_mfcc_mean8', 'first_mfcc_mean9',
                'first_mfcc_mean10', 'first_mfcc_mean11', 'first_mfcc_mean12', 'first_mfcc_mean13',
                'first_mfcc_mean14', 'first_mfcc_mean15', 'first_mfcc_mean16', 'first_mfcc_mean17',
                'first_mfcc_mean18', 'first_mfcc_mean19', 'first_mfcc_mean20', 'first_mfcc_std1', 'first_mfcc_std2',
                'first_mfcc_std3', 'first_mfcc_std4', 'first_mfcc_std5', 'first_mfcc_std6', 'first_mfcc_std7',
                'first_mfcc_std8', 'first_mfcc_std9', 'first_mfcc_std10', 'first_mfcc_std11', 'first_mfcc_std12',
                'first_mfcc_std13', 'first_mfcc_std14', 'first_mfcc_std15', 'first_mfcc_std16', 'first_mfcc_std17',
                'first_mfcc_std18', 'first_mfcc_std19', 'first_mfcc_std20', 'first_cent_mean', 'first_cent_std',
                'first_cent_skew', 'first_rolloff_mean', 'first_rolloff_std',
                'second_mfcc_mean1', 'second_mfcc_mean2', 'second_mfcc_mean3', 'second_mfcc_mean4',
                'second_mfcc_mean5', 'second_mfcc_mean6', 'second_mfcc_mean7', 'second_mfcc_mean8', 'second_mfcc_mean9',
                'second_mfcc_mean10', 'second_mfcc_mean11', 'second_mfcc_mean12', 'second_mfcc_mean13',
                'second_mfcc_mean14', 'second_mfcc_mean15', 'second_mfcc_mean16', 'second_mfcc_mean17',
                'second_mfcc_mean18', 'second_mfcc_mean19', 'second_mfcc_mean20', 'second_mfcc_std1', 'second_mfcc_std2',
                'second_mfcc_std3', 'second_mfcc_std4', 'second_mfcc_std5', 'second_mfcc_std6', 'second_mfcc_std7',
                'second_mfcc_std8', 'second_mfcc_std9', 'second_mfcc_std10', 'second_mfcc_std11', 'second_mfcc_std12',
                'second_mfcc_std13', 'second_mfcc_std14', 'second_mfcc_std15', 'second_mfcc_std16', 'second_mfcc_std17',
                'second_mfcc_std18', 'second_mfcc_std19', 'second_mfcc_std20', 'second_cent_mean', 'second_cent_std',
                'second_cent_skew', 'second_rolloff_mean', 'second_rolloff_std']]



        from sklearn.model_selection import train_test_split


        # Разделение набора данных на тренировочные и тестовые наборы (train/test split)
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,
                                                                    random_state=17)
        # размер тренировочных и тестовых наборов (количество строк, колонок):
        print ('Train set:', X_train.shape, y_train.shape)
        print ('Test set:', X_test.shape, y_test.shape)



        from sklearn import svm
        clf = svm.SVC(kernel='rbf')   #  функция ядра - RBF (радиальная базисная функция)
        clf.fit(X_train, y_train)     # Обучение модели на тренировочном наборе
        model_params = clf.get_params()
        yhat = clf.predict(X_test)    # для прогнозирования новых значений:
        print("Prediction:", yhat[0:20])
        print("Real Value:", y_test[0:20])
        # Вычисление accuracy
        from sklearn import metrics
        print("Train set Accuracy: ", metrics.accuracy_score(y_train, clf.predict(X_train)))
        print("Test set Accuracy: ",metrics.accuracy_score(y_test, yhat) )

        wandb.init(project='speaker_validator', config=model_params)

        wandb.config.update({"test_size": 0.095,
                             "train_len": X_train.shape,
                             "test_len":  X_test.shape})

        from sklearn.metrics import classification_report,confusion_matrix

        print('CONFUSION_MATRIX :\n')
        print(confusion_matrix(y_test,yhat))
        print('\n')
        print('REPORT :\n')
        print(classification_report(y_test,yhat))
        self.__model = clf


    def predict(self, file1, file2):
        """Метод для сравнения 2-х аудиофуйлов"""
        clf = self.__model
        files =[file1, file2]
        features = []
        for file in files:
            name = file
            y, sr = librosa.load(name, mono=True, duration=5)
            features.extend([np.mean(e) for e in librosa.feature.mfcc(y=y, sr=sr,
                                                                      n_mfcc=20)])  # mfcc_mean<0..20>
            features.extend([np.std(e) for e in librosa.feature.mfcc(y=y, sr=sr,
                                                                     n_mfcc=20)])  # mfcc_std
            features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                                    axis=0)[0])  # cent_mean
            features.append(np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                                   axis=0)[0])  # cent_std
            features.append(scipy.stats.skew(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                                             axis=0)[0])  # cent_skew
            features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                                    axis=0)[0])  # rolloff_mean
            features.append(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                                   axis=0)[0])  # rolloff_std
        features = np.array([features])
        return bool(clf.predict(features)[0])



