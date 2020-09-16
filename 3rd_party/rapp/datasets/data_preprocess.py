#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
import pandas as pd
import numpy as np
import os


def make_folder(root, folder_name):
    if '/' in folder_name:
        folder_name = folder_name.replace('/', '')
    if 'processed' not in os.listdir(f'{root}/{folder_name}'):
        os.system(f'mkdir {root}/{folder_name}/processed')


def steel_preprocess(data_dir, folder_name, file_name, root):
    make_folder(root, folder_name)
    # directories
    load_dir = os.path.join(root, folder_name, 'raw', file_name)
    pro_save_dir = os.path.join(root, folder_name, 'processed', file_name)
    
    # read and save raw file
    steel = pd.read_csv(load_dir)

    # preprocess
    label_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    labels = []
    for i in steel[label_columns].values:
        labels += [np.argmax(i)]
    steel = pd.concat([steel.drop(label_columns, axis=1), pd.DataFrame({'label':labels})], axis=1)
    
    # save processed file
    steel.to_csv(pro_save_dir, index=False)


def cnc_mf_preprocess(data_dir, folder_name, file_name, root):
    make_folder(root, folder_name)
    # directories
    load_dir = os.path.join(root, folder_name, 'raw', file_name)
    pro_save_dir = os.path.join(root, folder_name, 'processed', file_name)

    # for label
    labels = {
            1: 'yes',
            2: 'yes',
            3: 'yes',
            4: 'no',
            5: 'no',
            6: 'yes',
            7: 'no',
            8: 'yes',
            9: 'yes',
            10: 'yes',
            11: 'yes',
            12: 'yes',
            13: 'yes',
            14: 'yes',
            15: 'yes',
            16: 'no',
            17: 'yes',
            18: 'yes'
    }

    # read and save raw file
    df_list = []
    label_list = []
    for i in range(1,19):
        tmp = pd.read_csv(f'{load_dir}/experiment_{i:02}.csv')
        label_list += [pd.DataFrame({'label':[labels[i]]*len(tmp)})]
        df_list += [tmp]
    cnc = pd.concat(df_list).reset_index(drop=True)
    labels = pd.concat(label_list).reset_index(drop=True)

    # save processed file
    cnc = pd.concat([
            cnc.drop(['Machining_Process'], axis=1),
            pd.get_dummies(cnc.Machining_Process)
        ], axis=1)
    labels.label = labels.label.map({'yes':0, 'no':1})
    pd.concat([cnc, labels], axis=1).to_csv(pro_save_dir, index=False)


def cnc_pvi_preprocess(data_dir, folder_name, file_name, root):
    make_folder(root, folder_name)
    # directories
    load_dir = os.path.join(root, folder_name, 'raw', file_name)
    pro_save_dir = os.path.join(root, folder_name, 'processed', file_name)

    # for label
    labels = {
        1:'yes',
        2:'yes',
        3:'yes',
        6:'no',
        8:'no',
        9:'no',
        10:'no',
        11:'yes',
        12:'yes',
        13:'yes',
        14:'yes',
        15:'yes',
        17:'yes',
        18:'yes'
    }
    null_label = [4, 5, 7, 16]

    # read and save raw file
    df_list = []
    label_list = []
    for i in labels.keys():
        tmp = pd.read_csv(f'{load_dir}/experiment_{i:02}.csv')
        label_list += [pd.DataFrame({'label':[labels[i]]*len(tmp)})]
        df_list += [tmp]
    cnc = pd.concat(df_list).reset_index(drop=True)
    labels = pd.concat(label_list).reset_index(drop=True)
    
    # save processed file
    cnc = pd.concat([
            cnc.drop(['Machining_Process'], axis=1),
            pd.get_dummies(cnc.Machining_Process)
        ], axis=1)
    # cnc.drop(['Machining_Process'], axis=1, inplace=True)
    labels.label = labels.label.map({'yes':0, 'no':1})
    pd.concat([cnc, labels], axis=1).to_csv(pro_save_dir, index=False)


def eo_preprocess(data_dir, folder_name, file_name, root):
    import itertools
    make_folder(root, folder_name)
    # directories
    load_dir = os.path.join(root, folder_name, 'raw', file_name)
    pro_save_dir = os.path.join(root, folder_name, 'processed', file_name)

    # read and save raw file
    df_list = [] 
    for run_mode, anom_exist in itertools.product(['optimized', 'standard'], ['normal', 'anomalous']):
        df = pd.read_csv(f'{data_dir}/{folder_name}/HRSS_{anom_exist}_{run_mode}.csv')

        df.insert(loc=2, value=run_mode, column='Run_Mode')
        df.insert(loc=3, value=anom_exist, column='Anom_Exist')
        
        df_list.append(df)

    eo = pd.concat(df_list).reset_index(drop=True)

    # preprocess
    run_map = {'optimized':0, 'standard':1}
    anom_map = {'normal':0, 'anomalous':1}
    eo.Run_Mode = eo.Run_Mode.map(run_map)
    eo.Anom_Exist = eo.Anom_Exist.map(anom_map)

    eo.drop(['Timestamp'], axis=1, inplace=True)
    label = eo['Labels']
    features = eo.drop(['Labels'], axis=1)

    eo = pd.concat([features, label], axis=1)
    # save processed file
    eo.to_csv(pro_save_dir, index=False)


def nasa_preprocess(data_dir, folder_name, file_name, root):
    make_folder(root, folder_name)
    # directories
    load_dir = os.path.join(root, folder_name, 'raw', file_name)
    pro_save_dir = os.path.join(root, folder_name, 'processed', file_name)

    # read and save raw file
    nasa = pd.read_csv(load_dir)

    # preprocess
    drop_cols = [
        'Neo Reference ID', 
        'Name', 
        'Close Approach Date', 
        'Orbit Determination Date', 
        'Equinox', 
        'Orbiting Body'
    ]
    nasa.drop(drop_cols, axis=1, inplace=True)

    nasa.Hazardous = nasa.Hazardous.map(int)
    
    # save processed file
    nasa.to_csv(pro_save_dir, index=False)


def otto_preprocess(data_dir, folder_name, file_name, root):
    make_folder(root, folder_name)
    # directories
    load_dir = os.path.join(root, folder_name, 'raw', file_name)
    pro_save_dir = os.path.join(root, folder_name, 'processed', file_name)

    # read and save raw file
    otto = pd.read_csv(load_dir)

    # preprocess
    otto.drop(['id'], axis=1, inplace=True)
    label_map = {
        'Class_1': 0,
        'Class_2': 1,
        'Class_3': 2,
        'Class_4': 3,
        'Class_5': 4,
        'Class_6': 5,
        'Class_7': 6,
        'Class_8': 7,
        'Class_9': 8
    }
    otto.target = otto.target.map(label_map)
    # save processed file
    otto.to_csv(pro_save_dir, index=False)


def get_preprocess(data_name, data_config, root):
    data_dir = 'datasets/data/'

    if data_name == 'steel':
        return steel_preprocess(data_dir, data_config['folder_name'], data_config['file_name'], root)
    elif data_name == 'cnc_mf':
        return cnc_mf_preprocess(data_dir, data_config['folder_name'], data_config['file_name'], root)
    elif data_name == 'cnc_pvi':
        return cnc_pvi_preprocess(data_dir, data_config['folder_name'], data_config['file_name'], root)
    elif data_name == 'eo':
        return eo_preprocess(data_dir, data_config['folder_name'], data_config['file_name'], root)
    elif data_name == 'nasa':
        return nasa_preprocess(data_dir, data_config['folder_name'], data_config['file_name'], root)
    elif data_name == 'card':
        return card_preprocess(data_dir, data_config['folder_name'], data_config['file_name'], root)
    elif data_name == 'otto':
        return otto_preprocess(data_dir, data_config['folder_name'], data_config['file_name'], root)
    else:
        raise NotImplementedError(f'there is no predefined preprocess def with dataset {data_name}')


def get_data_from_url(url, file_name, root):
    if file_name not in os.listdir(root):
        os.system(f'wget -P {root} {url}')
    if 'zip' in file_name:
        os.system('unzip {root}/{file_name} -d {root}/')