#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:56:33 2018

@author: ezugasti
"""

import pandas as pd
import glob
import os
import time
import async_func
from tqdm import tqdm

month_set = set()


# Honek ez dau ezer rarua eitten, ondo dagola suposatzen dot

def convertXLSXtoCSV(datadir, savedir):
    for i in glob.glob(datadir + 'Data*.xlsx'):
        path, filename = os.path.split(i)  # split path/file
        # Read Excel file
        t = time.time()  # start monitoring time (usually this is slow...)
        data = pd.read_excel(i)  # read excel
        print('readTime ' + str(time.time() - t))  # print elapsed
        # Write to CSV
        t = time.time()
        newFilename = os.path.splitext(filename)[0] + '.csv'
        data.to_csv(savedir + newFilename, index=False)  # Write Excel
        print('writeTime ' + str(time.time() - t))


def createBatchCSV(datadir, savedir):
    for i in glob.glob(datadir + 'Data*.csv'):
        print(i)
        data = pd.read_csv(i)
        grouped = data.groupby('Batch')
        for name, group in grouped:
            if not ('FL' in name):
                continue
            filename = name[2:] + '.csv'  # remove 'FL' from batchname and add csv filetype
            if os.path.isfile(savedir + filename):  # check if CSV in Batch saved folder exists
                prevdata = pd.read_csv(savedir + filename)  # if so,load data
                info = pd.concat([prevdata, group])  # and concatenate tables
            else:
                info = group  # if not, create new dataframe with data
            debug_func(info, filename)
            info.to_csv(savedir + filename, index=False)


# ------------------------- honaino aldatu barik  --------------------------

def get_line_number(row):
    '''
    Checks VOR, INSM1 and INSM2 values, and returns the line number depending on this
    :param row:
    :return:
    '''
    ret = None
    if row['INSMA1: ID'] != 0 and row['INSMA2: ID'] != 0:
        if row['VORMA: ID'] != 0:
            ret = 1
        elif row['VORMB: ID'] != 0:
            ret = 2
    elif row['INSMB1: ID'] != 0 and row['INSMB2: ID'] != 0:
        if row['VORMA: ID'] != 0:
            ret = 3
        elif row['VORMB: ID'] != 0:
            ret = 4

    return ret


def filter_invalid(df):
    '''
    Filters every row where VORA=VORB=0, INSMA1=INSMB1 or INSMA2=INSMB2
    :param df:
    :return: filtered dataframe
    '''
    df = df[(df['VORMA: ID'] != 0) | (df['VORMB: ID'] != 0)]
    df = df[(df['INSMA1: ID'] != 0) | (df['INSMB1: ID'] != 0)]
    df = df[(df['INSMA2: ID'] != 0) | (df['INSMB2: ID'] != 0)]

    return df


def set_valid_line_numbers(data):
    '''
    Filters invalid rows and adds lineNumber values to each row.
    :param data:
    :return: data
    '''
    data = filter_invalid(data)
    data['lineNumber'] = data.apply(get_line_number, axis=1)
    return data


def set_good_names(data):
    '''
    Renames wrongly named columns
    :param data:
    :return:
    '''
    dict = {"INSMA1: Pos_9_Opr": "INSMA1: Oprek P9", "INSMA2: POS9_Opr": "INSMA2: Oprek_P9",
            'INSMA2: AP_P_Sta': 'INSMA2: AP_P_Stat', 'INSMA2: AP_V_Sta': 'INSMA2: AP_V_Stat', }
    data = data.rename(columns=dict)
    return data


def translate_english_columns(data):
    '''
    Renames english named columns to dutch
    :param data:
    :return:
    '''
    trans = {"Index": "Doorzet", "Holder": "Houder", "Rejects": "Uitval", "SP": "Halff."}

    for key, value in trans.items():
        data.columns = data.columns.str.replace(key, value)
    return data


def rename(x):
    '''
    Returns VORMB, INSMB1, INSMB2, VORMA, INSMA1, INSMA2 without the A or B letters
    VORMB => VORM, INSMB1 => INSM1
    :param x:
    :return:
    '''
    key, var = x.split(': ')
    if 'A' in key:
        return key.replace("A", "") + ': ' + var
    else:
        return key.replace("B", "") + ': ' + var


def debug_func(data, filename):
    cols = data.columns.values

    if 'MEET: Pubo2-RM1' in cols:
        print(filename, 'badao')
    else:
        print(filename, 'eztao')
    print(data.shape)


def checkChain(data, headers):
    for header in headers:
        grouped = data.groupby(header)
        keypart = header.split(': ')[0]
        ndata = pd.DataFrame()
        for k in grouped.groups:
            gr = grouped.get_group(k)
            if k == 0:  # data from Chain A
                filter_col = [col for col in gr if col.startswith(keypart)]
                gr = gr.drop(filter_col, axis=1)  # drop values from chain B, assets dont pass over this chain
                cols = pd.Series(gr.columns)
                keypart2 = keypart.replace("B", 'A')
                cols = cols.apply(lambda x: rename(x) if keypart2 in x else x)  # change name, delete A or B
                gr.columns = cols
                head2 = {keypart2.replace("A", "") + ': ID': 0}
                gr = deleteErrors(gr, head2)

            else:  # Data from Chain B
                keypart2 = keypart.replace("B", 'A')
                filter_col = [col for col in gr if col.startswith(keypart2)]
                gr = gr.drop(filter_col, axis=1)  # drop values from chain A, assets dont pass over this chain
                cols = pd.Series(gr.columns)
                cols = cols.apply(lambda x: rename(x) if keypart in x else x)  # change name, delete A or B
                gr.columns = cols
            if ndata.empty:
                ndata = gr
            else:
                ndata = pd.concat([ndata, gr], join='inner')
        data = ndata
    return data


def deleteErrors(data, headers):
    for key, value in headers.items():
        if 'Profil' in key:
            continue
        ndata = pd.DataFrame()
        grouped = data.groupby(key)
        for k in grouped.groups:
            gr = grouped.get_group(k)
            if k != value:
                if ndata.empty:
                    ndata = gr
                else:
                    ndata = pd.concat([ndata, gr], join='inner')
        data = ndata
    return data


def set_line_numbers(csv_file):
    path, filename = os.path.split(csv_file)
    data = pd.read_csv(csv_file)  # read CSV

    data = set_valid_line_numbers(data)  # Invalid data is dropped and line numbers are assigned
    data = set_good_names(data)  # The paired columns that are not well written are corrected

    data = translate_english_columns(data)
    data = checkChain(data, {'INSMB1: ID': 0, 'INSMB2: ID': 0, 'VORMB: ID': 0})  # Paired columns are merged

    headers = {col: 0 for col in data if col.endswith(': ID') and (col[:4] not in ['VORM', 'INSM', 'Prof'])}
    # data = deleteErrors(data, headers)  # Not paired columns are checked

    data.to_csv(savedir + filename, index=False)


def cleanCSV(datadir, savedir):
    files = glob.glob(datadir + '*.csv')

    async_func.parallel_process(files, set_line_numbers)


def NumericCSV(datadir, savedir):
    colsToDelete = pd.read_csv(datadir + '../labelData.csv', header=None)
    #    colsToDelete.iloc[0]=['','']
    csv_list = glob.glob(datadir + '*.csv')

    arr = [{"csv_file": file, 'colsToDelete': colsToDelete, 'savedir': savedir} for file in csv_list]
    async_func.parallel_process(arr, delete_non_numeric_columns, use_kwargs=True)


def delete_non_numeric_columns(csv_file, colsToDelete, savedir):
    path, filename = os.path.split(csv_file)
    batch = int(filename[0:-4])
    data = pd.read_csv(csv_file)  # read CSV

    if data.empty:
        return None
    try:
        if batch < 601027:
            data = data.drop(colsToDelete[0].values, axis=1)
        else:
            data = data.drop(colsToDelete[1].values, axis=1)
        data = data.sort_index(axis=1)
        print(str(batch) + ' ok')
    except Exception as e:
        print(str(batch) + ' NOK', e.args)
        return None
    #        data['Time'] = pd.to_datetime(data['Time'])
    #        data=data.set_index(data['Time'])
    # data = data.drop(['Time'], axis=1) # This is not used in order to predict data related values
    data.to_csv(savedir + filename, index=False)


def CommonHeader(datadir, savedir):
    for idx, i in enumerate(glob.glob(datadir + '*.csv')):
        path, filename = os.path.split(i)
        #        batch=int(filename[0:-4])
        data = pd.read_csv(i)  # read CSV

        if idx == 0:
            columns = data.columns.tolist()
            d_orig = data
        else:
            cols = data.columns.tolist()
            column_len = len(columns)
            col_len = len(cols)
            if column_len < col_len:
                result = list(set(cols) - set(columns))
                data = data.drop(result, axis=1)
                cols = data.columns.tolist()
            elif column_len > col_len:
                result = list(set(columns) - set(cols))
                data[result] = d_orig[result]
                cols = data.columns.tolist()

            data.columns = columns
            data.to_csv(savedir + filename, index=False)


def get_defect_data():
    base_dir = get_data_dir('OriginalCSV')
    data = pd.read_csv(base_dir + 'Defects per batch May 2018.csv')
    return data


def checkBatchMulticlass(batch, batches_w_defect):
    for type, batch_w_defect in batches_w_defect.items():
        if batch in batch_w_defect:
            return type
    return 0


def set_status_multiclass(csv_file, batches_w_defect):
    data = pd.read_csv(csv_file)  # read CSV
    _, filename = os.path.split(csv_file)
    batch = int(filename[0:-4])
    status = checkBatchMulticlass(batch, batches_w_defect)
    data['Status'] = status
    return data


def get_ok_batches():
    data_c = get_defect_data()
    data_c['SUM'] = data_c['Defect 1'] + data_c['Defect 2'] + data_c['Defect 3']
    batch_ok = data_c['Batch'].where(data_c['SUM'] == 0).dropna()
    return list(map(int, batch_ok.values))


def get_batches_w_defect_by_type():
    data = get_defect_data()
    defects = dict()
    defects[1] = data['Batch'].where(data['Defect 1'] > 0).dropna()
    defects[1] = list(map(int, defects[1].values))
    defects[2] = data['Batch'].where(data['Defect 2'] > 0).dropna()
    defects[2] = list(map(int, defects[2].values))
    defects[3] = data['Batch'].where(data['Defect 3'] > 0).dropna()
    defects[3] = list(map(int, defects[3].values))

    return defects


def JoinCSVMUlticlass(datadir, savedir):
    defects = get_batches_w_defect_by_type()
    csv_list = glob.glob(datadir + '*.csv')

    array = [{"csv_file": csv, "batches_w_defect": defects} for csv in csv_list]
    data = async_func.parallel_process(array, set_status_multiclass, use_kwargs=True)

    hData = pd.concat(data)
    hData = hData.dropna(axis='columns')

    hData.to_csv(savedir + 'Complete.csv', index=False)


def JoinCSV(datadir, savedir):
    batches_ok = get_ok_batches()

    csv_list = glob.glob(datadir + '*.csv')

    array = [{"csv_file": csv, "batches_ok": batches_ok} for csv in csv_list]
    data = async_func.parallel_process(array, set_status, use_kwargs=True)

    hData = pd.concat(data)
    hData = hData.dropna(axis='columns')
    hData.to_csv(savedir + 'Complete.csv', index=False)


def set_status(csv_file, batches_ok):
    data = pd.read_csv(csv_file)  # read CSV
    _, filename = os.path.split(csv_file)
    batch = int(filename[0:-4])
    status = int(batch in batches_ok)
    data['Status'] = status
    return data


def filter_w_key_values(df, dict):
    for k, v in dict.items():
        df = df[df[k] != v]
    return df


def get_data_dir(dirname):
    return os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dirname, )), '')


if __name__ == "__main__":
    # datadir='/media/ezugasti/Data/Ezugasti/Dropbox/Dropbox (MGEP)/Philips Lighting/'
    # savedir='/media/ezugasti/Data/Ezugasti/Dropbox/Dropbox (MGEP)/Ezugasti/Productive/philips/processedData/originalCSV/'
    datadir = '/media/ezugasti/Data/Ezugasti/Dropbox/Dropbox (MGEP)/Ezugasti/Productive/philips/processedData/NewData/OriginalXLSX/'
    savedir = async_func.get_data_dir('OriginalCSV')
    # CSV reading is much faster... converting (only once)
    # convertXLSXtoCSV(datadir,savedir)
    datadir = savedir
    # savedir='/media/ezugasti/Data/Ezugasti/Dropbox/Dropbox (MGEP)/Ezugasti/Productive/philips/processedData/BatchCSV/'
    savedir = get_data_dir('BatchCSV')
    # Splitting CSVs by Batch Number (only once)
    # createBatchCSV(datadir, savedir)
    datadir = savedir
    savedir = get_data_dir('ChainCleanCSV')

    # cleanCSV(datadir, savedir)  # remove Data from other chains

    datadir = savedir
    savedir = get_data_dir('NumericalCSV')
    # NumericCSV(datadir, savedir)  # Keep only numerical data

    # create common headers
    # datadir = savedir
    # savedir='/media/ezugasti/Data/Ezugasti/Dropbox/Dropbox (MGEP)/Ezugasti/Productive/philips/processedData/CommonCSV/'
    # savedir = get_data_dir('CommonCSV')
    # CommonHeader(datadir, savedir)

    # create Huge Dataset for MachineLearning Classification
    # If there is an error in a batch, we will consider faulty the whole batch
    datadir = savedir
    savedir = get_data_dir('HugeCSV')
    JoinCSV(datadir, savedir)

    # create Huge Dataset for MachineLearning Classification
    # If there is an error in a batch, we will consider faulty the whole batch
    # Divide different types of faults
    savedir = get_data_dir('HugeCSVMulticlass')
    JoinCSVMUlticlass(datadir, savedir)
