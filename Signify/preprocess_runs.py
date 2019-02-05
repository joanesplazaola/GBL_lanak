"""
Created on 20/11/18
@author: joanes

"""

import pandas as pd
import numpy as np
import glob
import os
import async_func


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


def clean_defect_columns(datadir, savedir):
    csv_list = glob.glob(datadir + 'Data*.csv')
    params = [{'csv': file, 'savedir': savedir, } for file in csv_list]
    async_func.parallel_process(params, delete_defect_rows, use_kwargs=True)


def delete_defect_rows(csv, savedir):
    path, filename = os.path.split(csv)
    df = pd.read_csv(csv)
    filtered_data = df[pd.isna(df['Reject'])]
    filtered_data.to_csv(savedir + filename, index=False)


def NumericCSV(datadir, savedir):
    colsToDelete = pd.read_csv(datadir + '../labelData.csv', header=None)
    #    colsToDelete.iloc[0]=['','']
    csv_list = glob.glob(datadir + '*.csv')

    arr = [{"csv_file": file, 'colsToDelete': colsToDelete, 'savedir': savedir} for file in csv_list]
    async_func.parallel_process(arr, delete_non_numeric_columns, use_kwargs=True)


def delete_non_numeric_columns(csv_file, colsToDelete, savedir):
    path, filename = os.path.split(csv_file)
    data = pd.read_csv(csv_file)  # read CSV

    data = data.drop(colsToDelete[0].values, axis=1)
    data = data.sort_index(axis=1)

    data.to_csv(savedir + filename, index=False)


def get_defect_data():
    base_dir = async_func.get_data_dir('OriginalCSV')
    data = pd.read_csv(base_dir + 'Defects per batch May 2018.csv')
    return data


def checkBatchMulticlass(row, batches_w_defect):
    batch = int(row['Batch'][2:])
    for type, batch_w_defect in batches_w_defect.items():
        if batch in batch_w_defect:
            return type
    return 0


def set_status_multiclass(csv_file, batches_w_defect):
    data = pd.read_csv(csv_file)  # read CSV
    path, filename = os.path.split(csv_file)
    data['Status'] = data.apply(checkBatchMulticlass, batches_w_defect=batches_w_defect, axis=1)
    data = data.dropna(axis='columns')
    data.drop(columns=['Time', 'Batch'], inplace=True)
    data.to_csv(async_func.get_data_dir('RunStatusCSV') + filename, index=False)


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


def JoinCSVMUlticlass(datadir):
    defects = get_batches_w_defect_by_type()
    csv_list = glob.glob(datadir + '*.csv')

    array = [{"csv_file": csv, "batches_w_defect": defects} for csv in csv_list]
    async_func.parallel_process(array, set_status_multiclass, use_kwargs=True)


if __name__ == "__main__":
    datadir = async_func.get_data_dir('OriginalCSV')
    savedir = async_func.get_data_dir('RunDefectFreeCSV')
    # clean_defect_columns(datadir, savedir)

    datadir = savedir
    savedir = async_func.get_data_dir('RunChainCSV')

    # cleanCSV(datadir, savedir)  # remove Data from other chains

    datadir = savedir
    savedir = async_func.get_data_dir('RunNumericalCSV')
    #NumericCSV(datadir, savedir)  # Keep only numerical data

    JoinCSVMUlticlass(savedir)
