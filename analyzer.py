from pandas.io.parsers import read_csv
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
import json

import preProcessing

import tensorflow as tf
# GPUセットアップ RAM消費を抑えるやつ
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(
            device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

filepath_f = open('./filepath.json', 'r')
filepath_json_dict = json.load(filepath_f)

# df = pd.read_csv(
#     'D:/Users/poly_Z/Documents/splatmusicprj/battle-results-csv_connect/output.csv')
# print(df.head())
# print(df.shape)

# weaponNameDf = (pd.read_csv(
#     'D:/Users/poly_Z/Documents/splatmusicprj/battle-results-csv_connect/statink-weapon2.csv'))['key'].sort_values()
# print(weaponNameDf.head())
# print(weaponNameDf.shape)

# df2 = copy.deepcopy(
#     df.loc[
#         df['game-ver'].isin(
#             [
#                 '5.5.0',
#                 '5.4.0',
#                 '5.3.1',
#                 '5.3.0',
#                 '5.2.2',
#                 '5.2.1',
#                 '5.2.0',
#                 '5.1.0',
#                 '5.0.1',
#                 '5.0.0',
#                 # '4.9.1', '4.9.0', '4.8.0', '4.7.0', '4.6.1', '4.6.0', '4.5.1', '4.5.0', '4.4.0', '4.3.1', '4.3.0', '4.2.0', '4.1.0', '4.0.0'
#             ]
#         ) & (
#             df['lobby-mode'].isin(['gachi'])
#         ),
#         [
#             'mode', 'stage', 'A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon', 'B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon', 'win'
#         ]
#     ]
# )

# # print(df.loc[(df['A1-weapon'] == "52gal_becchu")
# #              & (df['A2-weapon'] == "52gal_becchu")])
# # exit()
# # ***********************************************************************************************************

# # del df
# # df3 = copy.deepcopy(df2.dropna())
# # del df2
# # df4 = copy.deepcopy(pd.get_dummies(df3, columns=[
# #                     'mode', 'stage', 'A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon', 'B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon']))
# # del df3
# # df5 = copy.deepcopy(df4.replace({'win': {'alpha': 1, 'bravo': 0}}))
# # del df4
# # X = df5.loc[:, 'mode_area':'B4-weapon_wakaba'].values
# # y = df5.loc[:, 'win']

# # ***********************************************************************************************************
# # del df
# # df3 = copy.deepcopy(df2.dropna())
# # del df2
# # df4 = copy.deepcopy(pd.get_dummies(df3, columns=[
# #                     'mode', 'stage', 'A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon', 'B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon', 'win']))
# # # print(df3.head())
# # del df3
# # df5 = copy.deepcopy(df4)
# # del df4
# # X = df5.loc[:, 'mode_area':'B4-weapon_wakaba'].values
# # print(df5.head())
# # y = df5.loc[:, ['win_alpha', 'win_bravo']]

# # ***********************************************************************************************************
# # del df
# # df3 = copy.deepcopy(df2.dropna())
# # del df2
# # print(df3.head())
# # print(df3.shape)
# # modeAndStage = df3[['mode', 'stage']]
# # weaponA = df3[['A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon']]
# # weaponB = df3[['B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon']]
# # win = df3[['win']]
# # weaponA = np.sort(weaponA, axis=1)
# # weaponB = np.sort(weaponB, axis=1)
# # df4 = copy.deepcopy(
# #     pd.concat(
# #         [
# #             pd.DataFrame(modeAndStage, index=df3.index.values,
# #                          columns=['mode', 'stage']),
# #             pd.DataFrame(weaponA, index=df3.index.values,
# #                          columns=['A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon']),
# #             pd.DataFrame(weaponB, index=df3.index.values,
# #                          columns=['B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon']),
# #             pd.DataFrame(win, index=df3.index.values,
# #                          columns=['win'])
# #         ],
# #         axis=1
# #     )
# # )
# # print(df4.head())
# # print(df4.shape)
# # del df3, modeAndStage, weaponA, weaponB, win
# # df5 = copy.deepcopy(pd.get_dummies(df4, columns=[
# #                     'mode', 'stage', 'A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon', 'B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon', 'win']))
# # del df4
# # X = df5.loc[:, 'mode_area':'B4-weapon_wakaba'].values
# # # print(df5.head())
# # y = df5.loc[:, ['win_alpha', 'win_bravo']]
# # ***********************************************************************************************************

# del df
# df3 = copy.deepcopy(df2.dropna())
# del df2
# print(df3.head())
# print(df3.shape)
# modeAndStage = df3[['mode', 'stage']]
# win = df3[['win']]

# weaponCounterA = (
#     pd.DataFrame(pd.get_dummies(
#         df3[['A1-weapon']]).values, index=df3.index.values, columns=weaponNameDf) +
#     pd.DataFrame(pd.get_dummies(
#         df3[['A2-weapon']]).values, index=df3.index.values, columns=weaponNameDf) +
#     pd.DataFrame(pd.get_dummies(
#         df3[['A3-weapon']]).values, index=df3.index.values, columns=weaponNameDf) +
#     pd.DataFrame(pd.get_dummies(
#         df3[['A4-weapon']]).values, index=df3.index.values, columns=weaponNameDf)
# )/4
# weaponCounterA = weaponCounterA.rename(columns=lambda s: s+"A")
# print(weaponCounterA.max())

# weaponCounterB = (
#     pd.DataFrame(pd.get_dummies(
#         df3[['B1-weapon']]).values, index=df3.index.values, columns=weaponNameDf) +
#     pd.DataFrame(pd.get_dummies(
#         df3[['B2-weapon']]).values, index=df3.index.values, columns=weaponNameDf) +
#     pd.DataFrame(pd.get_dummies(
#         df3[['B3-weapon']]).values, index=df3.index.values, columns=weaponNameDf) +
#     pd.DataFrame(pd.get_dummies(
#         df3[['B4-weapon']]).values, index=df3.index.values, columns=weaponNameDf)
# )/4
# weaponCounterB = weaponCounterB.rename(columns=lambda s: s+"B")

# df4 = copy.deepcopy(
#     pd.concat(
#         [
#             pd.get_dummies(pd.DataFrame(modeAndStage, index=df3.index.values,
#                                         columns=['mode', 'stage']), columns=['mode', 'stage']),
#             weaponCounterA,
#             weaponCounterB,
#             pd.DataFrame(win, index=df3.index.values,
#                          columns=['win']).replace({'win': {'alpha': 1, 'bravo': 0}})
#         ],
#         axis=1
#     )
# )

# # ***********************************************************************************************************
# print(df4.head())
# print(df4.shape)
# del df3, modeAndStage, weaponCounterA, weaponCounterB, win
# df5 = copy.deepcopy(df4
#                     # pd.get_dummies(df4.replace(
#                     #     {'win': {'alpha': 1, 'bravo': 0}}), columns=['mode', 'stage'])
#                     )
# del df4
# print(df5)


# 下処理済データの生成も行うほう
df = preProcessing.preProcessing(
    filepath_json_dict['baseCsv'], filepath_json_dict['weaponNameList'])

# 既にある下処理済データを読み込む方
# df=pd.read_csv(filepath_json_dict['processedData'])

X = df.loc[:, 'mode_area':'wakabaB'].values
y = df.loc[:, 'win']

# ***********************************************************************************************************

diff = df.shape[1]-X.shape[1]
del df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# building the model
print('building the model ...')

model = Sequential()

model.add(Dense(X.shape[1]*15, input_shape=(X.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(2000))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(3000))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(2000))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(2000))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense((diff)))

rms = RMSprop()
model.compile(loss='mean_absolute_error', optimizer=rms, metrics=['accuracy'])

batch_size = 1024
nb_epoch = 100
# training
hist = model.fit(X_train, y_train,
                 batch_size=batch_size,
                 verbose=1,
                 epochs=nb_epoch,
                 validation_data=(X_test, y_test))

print(model.evaluate(X_test, y_test))

# model.save(filepath_json_dict['outputModel'])
model.save("D:/Users/poly_Z/Documents/splatmusicprj/analyzer/dust.h5")
