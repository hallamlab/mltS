'''
This file preprocesses the input data in PathoLogic File Format (.pf).
'''

import os
import os.path
import pickle as pkl
import sys
import time
import traceback
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from dataset_builder.build_dataset import map_labels2functions, build_minpath_dataset
from feature_builder.features_helper import build_features_matrix
from utility.access_file import save_data, load_data


# EXTRACT INFORMATION FROM EXPERIMENTAL DATASET -------

def __parse_input(input_path):
    """ Process input from a given path
    :type input_path: str
    :param input_path: The RunPathoLogic input path, where all the data folders
        are located
    """

    for file_name in os.listdir(input_path):
        if file_name.endswith('.pf'):
            input_file = os.path.join(input_path, file_name)
            break
    if os.path.isfile(input_file):
        print('\t\t\t--> Prepossessing input file from: {0}'.format(input_file.split('/')[-1]))
        product_info = OrderedDict()
        with open(input_file, errors='ignore') as f:
            for text in f:
                if not str(text).startswith('#'):
                    ls = text.strip().split('\t')
                    if ls:
                        if ls[0] == 'ID':
                            product_id = ' '.join(ls[1:])
                            product_name = ''
                            product_type = ''
                            product_ec = ''
                        elif ls[0] == 'PRODUCT':
                            product_name = ' '.join(ls[1:])
                        elif ls[0] == 'PRODUCT-TYPE':
                            product_type = ' '.join(ls[1:])
                        elif ls[0] == 'EC':
                            product_ec = 'EC-'
                            product_ec = product_ec + ''.join(ls[1:])
                        elif ls[0] == '//':
                            # datum is comprised of {ID: (PRODUCT, PRODUCT-TYPE, EC)}
                            datum = {product_id: (product_name, product_type, product_ec)}
                            product_info.update(datum)
        return product_info


def __parse_output(output_path, pathway_id, use_idx_only=False):
    """ Process input from a given path
    :param idx:
    :type output_path: str
    :param output_path: The RunPathoLogic input path, where all the data folders
        are located
    """
    header = False
    ptwy_txt = ''
    ptwy_dat = ''
    ptwy_col = ''

    for file_name in os.listdir(output_path):
        if file_name.endswith('.txt'):
            ptwy_txt = os.path.join(output_path, file_name)
        elif file_name.endswith('.dat'):
            ptwy_dat = os.path.join(output_path, file_name)
        elif file_name.endswith('.col'):
            ptwy_col = os.path.join(output_path, file_name)

    if os.path.isfile(ptwy_txt):
        lst_pathways_idx = list()
        print('\t\t\t--> Prepossessing output file from: {0}'.format(ptwy_txt.split('/')[-1]))
        with open(ptwy_txt, errors='ignore') as f:
            for text in f:
                if not str(text).startswith('#'):
                    ls = text.strip().split('\t')
                    if ls:
                        if not header:
                            if ls[0] == 'SAMPLE':
                                header = True
                                ptwy_id = ls.index('PWY_NAME')
                        else:
                            if ls[ptwy_id]:
                                if ls[ptwy_id] in pathway_id:
                                    if use_idx_only:
                                        lst_pathways_idx.append(pathway_id[ls[ptwy_id]])
                                    else:
                                        lst_pathways_idx.append(ls[ptwy_id])

    else:
        lst_pathways_idx = list()
        if os.path.isfile(ptwy_dat):
            print('\t\t\t--> Prepossessing output file from: {0}'.format(ptwy_dat.split('/')[-1]))
            with open(ptwy_dat, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split()
                        if ls:
                            if ls[0] == 'UNIQUE-ID':
                                ptwy_id = ' '.join(ls[2:])
                                if ptwy_id in pathway_id:
                                    if ptwy_id not in lst_pathways_idx:
                                        if use_idx_only:
                                            lst_pathways_idx.append(pathway_id[ptwy_id])
                                        else:
                                            lst_pathways_idx.append(ptwy_id)

        if os.path.isfile(ptwy_col):
            print('\t\t\t--> Prepossessing output file from: {0}'.format(ptwy_col.split('/')[-1]))
            with open(ptwy_col, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split('\t')
                        if ls:
                            if not header:
                                if ls[0] == 'UNIQUE-ID':
                                    header = True
                                    for (i, item) in enumerate(ls):
                                        if item == 'UNIQUE-ID':
                                            pathway_idx = i
                                            break
                            else:
                                ptwy_id = ls[pathway_idx]
                                if ptwy_id in pathway_id:
                                    if ptwy_id not in lst_pathways_idx:
                                        if use_idx_only:
                                            lst_pathways_idx.append(pathway_id[ptwy_id])
                                        else:
                                            lst_pathways_idx.append(ptwy_id)
    return lst_pathways_idx


def __extract_features_from_input(input_path, idx, total_samples):
    '''

    :param input_path:
    :param idx:
    :param total_samples:
    :return:
    '''
    if os.path.isdir(input_path) or os.path.exists(input_path):
        print(
            '\t\t{1:d})- Progress ({0:.2f}%): extracted input information from {1:d} samples (out of {2:d})...'.format(
                (idx + 1) * 100.00 / total_samples, idx + 1, total_samples))

        # Preprocess inputs and outputs
        input_info = __parse_input(input_path=input_path)

        input_path = list()
        for i, item in input_info.items():
            if item[2]:
                input_path.append(item[2])
    else:
        print('\t>> Failed to preprocess {0} file...'.format(input_path.split('/')[-2]),
              file=sys.stderr)
    return input_path


def __extract_features_from_output(output_path, pathway_id, idx, total_samples):
    '''

    :param output_path:
    :param idx:
    :param total_samples:
    :return:
    '''
    if os.path.isdir(output_path) or os.path.exists(output_path):
        print(
            '\t\t{1:d})- Progress ({0:.2f}%): extracted output information from {1:d} samples (out of {2:d})...'.format(
                (idx + 1) * 100.00 / total_samples, idx + 1, total_samples))
        # Preprocess outputs
        output_path = __parse_output(output_path=output_path, pathway_id=pathway_id, use_idx_only=False)

    else:
        print('\t>> Failed to preprocess {0} file...'.format(output_path.split('/')[-2]),
              file=sys.stderr)
    return output_path


def __extract_input_from_pf_files(col_idx, col_id, folder_path, processes=2):
    lst_ipaths = [os.path.join(folder_path, folder) for folder in os.listdir(folder_path)
                  if not folder.startswith('.')]

    print('\t>> Extracting input information from {0} files...'.format(len(lst_ipaths)))

    pool = Pool(processes=processes)
    results = [pool.apply_async(__extract_features_from_input,
                                args=(lst_ipaths[idx], idx, len(lst_ipaths))) for
               idx in range(len(lst_ipaths))]
    output = [p.get() for p in results]

    X = np.zeros((len(output), len(col_idx)), dtype=np.int32)

    for idx, item in enumerate(output):
        for i in item:
            if i in col_id:
                t = np.where(col_idx == col_id[i])[0]
                X[idx, t] += 1
    return X


def __extract_output_from_pf_files(folder_path, pathway_id, processes=1):
    lst_opaths = [os.path.join(folder_path, folder) for folder in os.listdir(folder_path)
                  if not folder.startswith('.')]

    print('\t>> Extracting output information from {0} files...'.format(len(lst_opaths)))

    pool = Pool(processes=processes)
    results = [pool.apply_async(__extract_features_from_output,
                                args=(lst_opaths[idx], pathway_id, idx, len(lst_opaths))) for
               idx in range(len(lst_opaths))]
    output = [p.get() for p in results]

    y = np.empty((len(output),), dtype=np.object)

    for idx, item in enumerate(output):
        y[idx] = np.unique(item)
    sample_ids = [os.path.split(opath)[-1] for opath in lst_opaths]
    return y, sample_ids


# ---------------------------------------------------------------------------------------

def __parse_files(arg):
    ##########################################################################################################
    ######################            LOAD DATA OBJECT AND INDICATOR MATRICES           ######################
    ##########################################################################################################

    print('1)- Loading BIOCYC object from: {0:s}'.format(
        arg.ospath))
    data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
    row_tag = 'pathway'
    row_id = data_object['pathway_id']
    col_tag = 'ec'
    col_id = data_object['ec_id']
    if arg.construct_reaction:
        row_tag = 'reaction'
        row_id = data_object['reaction_id']
    if not arg.use_ec:
        col_tag = 'gene'
        col_id = data_object['gene_name_id']

    ptwy_ec_spmatrix, ptwy_ec_id = load_data(file_name=arg.pathway_ec, load_path=arg.ospath)

    ##########################################################################################################
    ######################      EXTRACTING INFORMATION FROM METEGENOMICS DATASET        ######################
    ##########################################################################################################

    print('\n2)- Extracting information from pf datasets...')
    if arg.ex_info_from_pf:
        X = __extract_input_from_pf_files(col_idx=ptwy_ec_id, col_id=col_id, folder_path=arg.inpath,
                                          processes=arg.n_jobs)
        num_samples = X.shape[0]
        file = arg.metegenomics_dataset + '_' + str(num_samples) + '_Xm.pkl'
        file_desc = '# The pf dataset representing a list of data components (X)...'
        save_data(data=file_desc, file_name=file, save_path=arg.dspath, tag='the pf dataset (X)',
                  mode='w+b')
        save_data(data=X, file_name=file, save_path=arg.dspath, mode='a+b', print_tag=False)

        y, sample_ids = __extract_output_from_pf_files(folder_path=arg.inpath, pathway_id=row_id,
                                                       processes=arg.n_jobs)
        file = arg.metegenomics_dataset + '_' + str(num_samples) + '_y.pkl'
        file_desc = '# The pf dataset representing a list of data components (y) with ids...'
        save_data(data=file_desc, file_name=file, save_path=arg.dspath, tag='the pf dataset (y)',
                  mode='w+b')
        save_data(data=(y, sample_ids), file_name=file, save_path=arg.dspath, mode='a+b', print_tag=False)

        build_minpath_dataset(X=X, col_idx=ptwy_ec_id, dict_id=col_id, num_samples=X.shape[0],
                              file_name=arg.metegenomics_dataset, save_path=arg.dspath)

    else:
        print('\t>> Extracting information from pf dataset is not indicated...')

    if arg.mapping:
        print('\n3)- Mapping labels with functions...')
        if y:
            map_labels2functions(row_data_matrix=ptwy_ec_spmatrix, col_postion_idx=ptwy_ec_id, y=y,
                                 num_samples=X.shape[0], col_id=col_id, row_id=row_id, map_all=arg.map_all,
                                 col_tag=col_tag, row_tag=row_tag, file_name=arg.metegenomics_dataset,
                                 save_path=arg.dspath)
        else:
            print('\t>> A list of vocab is not provided...')

    ##########################################################################################################
    ################################             BUILDING FEATURES             ###############################
    ##########################################################################################################

    print('\n4)- Extracting features from dataset...')
    if arg.ex_features_from_pf:
        if num_samples != 0:
            num_samples = y.shape[0]
        else:
            num_samples = 418

        f_name = os.path.join(arg.dspath, arg.ec_feature)
        print('\t\t## Loading the EC properties from: {0:s}'.format(f_name))
        with open(f_name, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    ec_properties_matrix = data
                    break

        f_name = os.path.join(arg.dspath, arg.metegenomics_dataset + '_' + str(num_samples) + '_Xm.pkl')
        print('\t\t## Loading dataset from: {0:s}'.format(f_name))
        with open(f_name, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    X = data
                    break

        feature_lst = [arg.num_reaction_evidence_features] + [arg.num_ec_evidence_features] + [
            arg.num_ptwy_evidence_features]
        matrix_list = [ptwy_ec_spmatrix] + [ec_properties_matrix]
        f_name = arg.metegenomics_dataset + '_' + str(num_samples)
        build_features_matrix(biocyc_object=data_object, X=X, matrix_list=matrix_list, col_idx=ptwy_ec_id,
                              features_list=feature_lst, display_interval=arg.display_interval,
                              constraint_kb=arg.constraint_kb, file_name=f_name, save_path=arg.dspath)
    else:
        print('\t>> Building features is not applied...')


def input_main(arg):
    try:
        if os.path.isdir(arg.ospath):
            timeref = time.time()
            print('*** PREPROCSSING PATHOLOGIC FORMATTED DATASETS...')
            __parse_files(arg)
            print('\n*** THE DATASET PROCESSING CONSUMED {0:f} SECONDS'.format(round(time.time() - timeref, 3)),
                  file=sys.stderr)
        else:
            print('\n*** PLEASE MAKE SURE TO PROVIDE THE CORRECT PATH FOR THE DATASETS '
                  'AND THE ASSOCIATED PARAMETERS', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
