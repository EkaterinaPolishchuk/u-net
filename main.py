import opendatasets as od
import pandas as pd
import numpy as np
from skimage.transform import resize
from skimage.io import imread
from sklearn.model_selection import train_test_split
from model import unet

'''
https://www.kaggle.com/code/inversion/run-length-decoding-quick-start/notebook
На вход маска, сжатая с помощью run-length encoding
Возвращает массив, в котором 1 - маска, 0 - фон
'''


def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# dice score
def mean_dice_score(y_true, y_pred):
    mean = 0.
    number_masks = y_true.shape[0]
    for i in range(number_masks):
        dice = 2 * np.sum(y_true[i] * y_pred[i]) / (np.sum(y_true[i]) + np.sum(y_pred[i]))
        mean += dice / number_masks
    return mean


if __name__ == "__main__":
    od.download("https://www.kaggle.com/competitions/airbus-ship-detection/data?select=train_v2")

    masks = pd.read_csv('./airbus-ship-detection/train_ship_segmentations_v2.csv')

    # только фотографии с кораблями
    masks = masks.loc[~pd.isna(masks['EncodedPixels'])].reset_index(drop=True)

    # заполняем массивы нулями
    X = np.zeros((110, 768, 768, 3), dtype=np.uint8)
    y = np.zeros((110, 768, 768, 1), dtype=np.bool)

    # список фотографий
    id_images = list(masks['ImageId'])

    # заполнение массива Х
    # только 110 фотографий, так как не хватает памяти
    for i, name in enumerate(id_images[:110]):
        path = './airbus-ship-detection/train_v2/' + name
        image = imread(path)
        image = resize(image, (768, 768), mode='constant', preserve_range=True)

        X[i] = image

    # заполнение массива у
    for i, name in enumerate(id_images[:110]):
        encodedPixels = masks.loc[masks['ImageId'] == name, 'EncodedPixels'].tolist()

        # заполняем массив нулями
        mask = np.zeros((768, 768))
        # создание массива масок
        for p in encodedPixels:
            mask += rle_decode(p)
            # добавляем ось, форма маски (768, 768, 1)
        mask = np.expand_dims(mask, axis=-1)

        y[i] = mask

    # разделяем на тестовый и тренировочный наборы
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train = X[:100]
    y_train = y[:100]
    X_test = X[100:]
    y_test = y[100:]

    # размеры фотографии
    IMG_WIDTH = 768
    IMG_HEIGHT = 768
    IMG_CHANNELS = 3

    model = unet(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

    # обучение
    model.fit(X_train, y_train, validation_split=0.1, batch_size=9, epochs=1)

    # прогноз
    y_pred = model.predict(X_test)

    # оценка на тестовых данных
    mean_dice = mean_dice_score(y_test, y_pred)
    print("Mean dice score:", mean_dice)
