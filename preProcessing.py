import numpy as np
import pandas as pd
import json

# 元データの下処理をして，DataFrameを返す関数
# 引数は元データのpathと，ブキ名リストがあるデータcsvのpath
# それぞれ下記からDL可能
# 元データ: https://dl-stats.stat.ink/splatoon-2/battle-results-csv/battle-results-csv.zip
# ブキ名: https://stat.ink/api/v2/weapon.csv


def preProcessing(dataFilePass: str, nameListFilePass: str):

    df = pd.read_csv(dataFilePass)
    print(df.head())
    print(df.shape)

    # ブキ名だけ欲しいので切り出し
    weaponNameDf = (pd.read_csv(nameListFilePass))['key'].sort_values()
    print(weaponNameDf.head())
    print(weaponNameDf.shape)

    # 元データをざっくり成形
    df = df.loc[

        # ゲーム側のバージョンの指定，ガチマッチのみのデータを使用
        df['game-ver'].isin(
            [
                '5.5.0',
                '5.4.0',
                '5.3.1',
                '5.3.0',
                '5.2.2',
                '5.2.1',
                '5.2.0',
                '5.1.0',
                '5.0.1',
                '5.0.0',
                # '4.9.1', '4.9.0', '4.8.0', '4.7.0', '4.6.1', '4.6.0', '4.5.1', '4.5.0', '4.4.0', '4.3.1', '4.3.0', '4.2.0', '4.1.0', '4.0.0'
            ]
        ) & (
            df['lobby-mode'].isin(['gachi'])
        ),
        # 試合前に分かる情報(ルール，ステージ，敵味方のブキ)のcolumnのみを使用+学習用に勝敗のcolumn(win)
        [
            'mode', 'stage', 'A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon', 'B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon', 'win'
        ]
    ]

    # NaNがあるrowを削除
    # これをしているので欠損アリのデータや回線落ちプレーヤーの居るデータは考慮していない
    df = df.dropna()

    # ブキデータを下処理したいので，それ以外(ルール，ステージ，勝敗)のcolumnは取り分けておく
    # もっといい方法があるかもしれないけど思いつかなかった
    modeAndStage = df[['mode', 'stage']]
    win = df[['win']]

    # 各プレーヤーのブキのデータをそれぞれone-hot化してから足し，その後正規化(チーム内で同名ブキは最大4件のため4で割っている)
    weaponCounterA = (
        pd.DataFrame(pd.get_dummies(
            df[['A1-weapon']]).values, index=df.index.values, columns=weaponNameDf) +
        pd.DataFrame(pd.get_dummies(
            df[['A2-weapon']]).values, index=df.index.values, columns=weaponNameDf) +
        pd.DataFrame(pd.get_dummies(
            df[['A3-weapon']]).values, index=df.index.values, columns=weaponNameDf) +
        pd.DataFrame(pd.get_dummies(
            df[['A4-weapon']]).values, index=df.index.values, columns=weaponNameDf)
    )/4
    # チームbravoに対してもalphaと同じ処理をするが，ブキ名のcolumnがかぶるためリネームしておく
    weaponCounterA = weaponCounterA.rename(columns=lambda s: s+"A")
    print(weaponCounterA.max())

    weaponCounterB = (
        pd.DataFrame(pd.get_dummies(
            df[['B1-weapon']]).values, index=df.index.values, columns=weaponNameDf) +
        pd.DataFrame(pd.get_dummies(
            df[['B2-weapon']]).values, index=df.index.values, columns=weaponNameDf) +
        pd.DataFrame(pd.get_dummies(
            df[['B3-weapon']]).values, index=df.index.values, columns=weaponNameDf) +
        pd.DataFrame(pd.get_dummies(
            df[['B4-weapon']]).values, index=df.index.values, columns=weaponNameDf)
    )/4
    weaponCounterB = weaponCounterB.rename(columns=lambda s: s+"B")

    # 取り分けておいたルール，ステージ，勝敗データとくっつける
    df = pd.concat(
        [
            pd.get_dummies(pd.DataFrame(modeAndStage, index=df.index.values,
                                        columns=['mode', 'stage']), columns=['mode', 'stage']),
            weaponCounterA,
            weaponCounterB,

            # 勝敗についてはalpha,bravo表記を1,0表記にしておく
            pd.DataFrame(win, index=df.index.values,
                         columns=['win']).replace({'win': {'alpha': 1, 'bravo': 0}})

            # one-hot化する場合はこっち
            # pd.get_dummies(pd.DataFrame(win, index=df.index.values,
            #             columns=['win']), columns=['win']),
        ],
        axis=1
    )

    # ***********************************************************************************************************
    # ローカルで動かしてるとRAMが死にかけたのでdel
    del modeAndStage, weaponCounterA, weaponCounterB, win

    print(df)
    return df


if __name__ == '__main__':
    filepath_f = open('./filepath.json', 'r')
    filepath_json_dict = json.load(filepath_f)
    data = filepath_json_dict['baseCsv']
    nameList = filepath_json_dict['weaponNameList']
    out = preProcessing(data, nameList)
    out.to_csv(
        filepath_json_dict['processedData'])
