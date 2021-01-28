__author__ = "Abdurrahman M. A. Basher"
__date__ = '08/12/2020'
__copyright__ = "Copyright 2020, The Hallam Lab"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Abdurrahman M. A. Basher"
__email__ = "ar.basher@alumni.ubc.ca"
__status__ = "Production"
__description__ = "This file is the main entry to perform learning and prediction on dataset using mltS model."

import datetime
import json
import os
import textwrap
from argparse import ArgumentParser

import utility.file_path as fph
from train import train
from utility.arguments import Arguments


def __print_header():
    os.system('clear')
    print('# ' + '=' * 50)
    print('Author: ' + __author__)
    print('Copyright: ' + __copyright__)
    print('License: ' + __license__)
    print('Version: ' + __version__)
    print('Maintainer: ' + __maintainer__)
    print('Email: ' + __email__)
    print('Status: ' + __status__)
    print('Date: ' + datetime.datetime.strptime(__date__,
                                                "%d/%m/%Y").strftime("%d-%B-%Y"))
    print('Description: ' + textwrap.TextWrapper(width=45,
                                                 subsequent_indent='\t     ').fill(__description__))
    print('# ' + '=' * 50)


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def __internal_args(parse_args):
    arg = Arguments()

    ###***************************         Global arguments         ***************************###
    arg.display_interval = parse_args.display_interval
    if parse_args.display_interval < 0:
        arg.display_interval = 1
    arg.random_state = parse_args.random_state
    arg.num_jobs = parse_args.num_jobs
    arg.num_models = parse_args.num_models
    arg.batch = parse_args.batch
    arg.max_inner_iter = parse_args.max_inner_iter
    arg.num_epochs = parse_args.num_epochs
    arg.shuffle = parse_args.shuffle

    ###***************************          Path arguments          ***************************###
    arg.ospath = parse_args.ospath
    arg.dspath = parse_args.dspath
    arg.mdpath = parse_args.mdpath
    arg.rspath = parse_args.rspath
    arg.rsfolder = parse_args.rsfolder
    arg.logpath = parse_args.logpath

    ###***************************          File arguments          ***************************###
    arg.object_name = parse_args.object_name
    arg.pathway2ec_name = parse_args.pathway2ec_name
    arg.pathway2ec_idx_name = parse_args.pathway2ec_idx_name
    arg.features_name = parse_args.features_name
    arg.hin_name = parse_args.hin_name
    arg.pi_name = parse_args.pi_name
    arg.X_name = parse_args.X_name
    arg.y_name = parse_args.y_name
    arg.source_name = parse_args.source_name
    arg.X_trust_name = parse_args.X_trust_name
    arg.y_trust_name = parse_args.y_trust_name
    arg.similarity_name = parse_args.similarity_name
    arg.vocab_name = parse_args.vocab_name
    arg.file_name = parse_args.file_name
    arg.samples_ids = parse_args.samples_ids
    arg.model_name = parse_args.model_name
    arg.dsname = parse_args.dsname

    ###***************************     Preprocessing arguments      ***************************###

    arg.preprocess_dataset = parse_args.preprocess_dataset
    arg.test_size = parse_args.test_size
    if arg.test_size < 0 or arg.test_size > 1:
        arg.test_size = 0
    arg.binarize_input_feature = parse_args.binarize
    arg.normalize_input_feature = parse_args.normalize
    arg.use_external_features = parse_args.use_external_features
    arg.cutting_point = parse_args.cutting_point

    ###***************************        Training arguments        ***************************###

    arg.train = parse_args.train
    arg.fit_intercept = parse_args.fit_intercept
    arg.train_selected_sample = parse_args.train_selected_sample
    arg.ssample_input_size = parse_args.ssample_input_size
    arg.ssample_mini_input_size = parse_args.ssample_mini_input_size
    arg.ssample_label_size = parse_args.ssample_label_size
    arg.calc_subsample_size = parse_args.calc_subsample_size
    arg.calc_label_cost = parse_args.calc_label_cost
    arg.calc_total_cost = parse_args.calc_total_cost
    arg.label_closeness_sim = parse_args.label_closeness_sim
    arg.corr_label_sim = parse_args.corr_label_sim
    arg.corr_input_sim = parse_args.corr_input_sim
    arg.early_stop = parse_args.early_stop
    arg.loss_threshold = parse_args.loss_threshold

    # apply trusted reference datasets
    arg.use_trusted = parse_args.use_trusted
    arg.beta_source = parse_args.beta_source
    arg.apply_trusted_loss = parse_args.apply_trusted_loss

    # apply peer learning
    arg.apply_peer = parse_args.apply_peer
    arg.beta_peer = parse_args.beta_peer

    # partial labeling
    arg.apply_partial_label = parse_args.apply_partial_label
    arg.use_trusted_partial_label = parse_args.use_trusted_partial_label
    arg.tau_partial = parse_args.tau_partial
    arg.alpha_partial = parse_args.alpha_partial
    arg.num_neighbors = parse_args.num_neighbors

    # apply hyperparameters
    arg.sigma = parse_args.sigma

    # apply regularization
    arg.alpha_glob = parse_args.alpha_glob
    arg.penalty = parse_args.penalty
    arg.fuse_weight = parse_args.fuse_weight
    arg.alpha_elastic = parse_args.alpha_elastic
    arg.l1_ratio = 1 - parse_args.l2_ratio
    arg.lambdas = parse_args.lambdas
    arg.kappa = parse_args.kappa

    # apply learning hyperparameter
    arg.learning_type = parse_args.learning_type
    arg.lr = parse_args.lr
    arg.lr0 = parse_args.lr0
    arg.forgetting_rate = parse_args.fr
    arg.delay_factor = parse_args.delay

    ###***************************       Prediction arguments       ***************************###
    arg.evaluate = parse_args.evaluate
    arg.predict = parse_args.predict
    arg.pathway_report = parse_args.pathway_report
    arg.extract_pf = True
    if parse_args.no_parse:
        arg.extract_pf = False
    arg.build_features = True
    if parse_args.no_build_features:
        arg.build_features = False
    arg.top_k_taxa = parse_args.top_k_taxa
    arg.plot = parse_args.plot
    arg.meta_predict = parse_args.meta_predict
    arg.meta_adaptive = parse_args.meta_adaptive
    arg.meta_omega = parse_args.meta_omega
    arg.pref_model = parse_args.pref_model
    arg.decision_threshold = parse_args.decision_threshold
    arg.soft_voting = parse_args.soft_voting
    arg.pref_rank = parse_args.pref_rank
    arg.top_k_rank = parse_args.top_k_rank
    arg.estimate_prob = parse_args.estimate_prob
    arg.apply_tcriterion = parse_args.apply_tcriterion
    arg.adaptive_beta = parse_args.adaptive_beta
    arg.psp_k = parse_args.psp_k
    return arg


def parse_command_line():
    __print_header()
    parser = ArgumentParser(description="Run mltS.")

    ###***************************         Global arguments         ***************************###
    parser.add_argument('--display-interval', default=1, type=int,
                        help='display intervals. -1 means display per each iteration. (default value: 1).')
    parser.add_argument('--random_state', default=12345, type=int, help='Random seed. (default value: 12345).')
    parser.add_argument('--num-jobs', type=int, default=2, help='Number of parallel workers. (default value: 2).')
    parser.add_argument('--num-models', default=3, type=int, help='Number of models to generate. (default value: 3).')
    parser.add_argument('--batch', type=int, default=500, help='Batch size. (default value: 500).')
    parser.add_argument('--max-inner-iter', default=15, type=int,
                        help='Number of inner iteration for logistic regression. (default value: 15)')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of epochs over the training set. (default value: 3).')
    parser.add_argument('--shuffle', action='store_false', default=True,
                        help='Whether or not the training data should be shuffled after each epoch. '
                             '(default value: True).')

    ###***************************          Path arguments          ***************************###
    parser.add_argument('--ospath', type=str, default=fph.OBJECT_PATH,
                        help='The path to the data object that contains extracted '
                             'information from the MetaCyc database. The default is '
                             'set to object folder outside the source code.')
    parser.add_argument('--dspath', type=str, default=fph.DATASET_PATH,
                        help='The path to the dataset after the samples are processed. '
                             'The default is set to dataset folder outside the source code.')
    parser.add_argument('--mdpath', type=str, default=fph.MODEL_PATH,
                        help='The path to the output models. The default is set to '
                             'train folder outside the source code.')
    parser.add_argument('--rspath', type=str, default=fph.RESULT_PATH,
                        help='The path to the results. The default is set to result '
                             'folder outside the source code.')
    parser.add_argument('--rsfolder', type=str, default="mltS_B2_symbionts",
                        help='The result folder name. The default is set to Prediction_mltS.')
    parser.add_argument('--logpath', type=str, default=fph.LOG_PATH,
                        help='The path to the log directory.')

    ###***************************          File arguments          ***************************###
    parser.add_argument('--object-name', type=str, default='biocyc.pkl',
                        help='The biocyc file name. (default value: "biocyc.pkl")')
    parser.add_argument('--pathway2ec-name', type=str, default='pathway2ec.pkl',
                        help='The pathway2ec association matrix file name. (default value: "pathway2ec.pkl")')
    parser.add_argument('--pathway2ec-idx-name', type=str, default='pathway2ec_idx.pkl',
                        help='The pathway2ec association indices file name. (default value: "pathway2ec_idx.pkl")')
    parser.add_argument('--features-name', type=str, default='path2vec_embeddings.npz',
                        help='The features file name. (default value: "path2vec_embeddings.npz")')
    parser.add_argument('--hin-name', type=str, default='hin.pkl',
                        help='The hin file name. (default value: "hin.pkl")')
    parser.add_argument('--pi-name', type=str, default='trans_prob.pkl',
                        help='The pi file name to construct transition probability matrix. (default value: "trans_prob.pkl")')
    parser.add_argument('--X-name', type=str, default='biocyc21_tier3_9392_Xe.pkl',
                        help='The X file name. (default value: "biocyc21_tier3_9392_Xe.pkl")')
    parser.add_argument("--y-name", nargs="+", type=str, default=["biocyc21_tier3_9392_y.pkl",
                                                                  "biocyc21_tier3_9392_minpath_y.pkl",
                                                                  "biocyc21_tier3_9392_mllr_y.pkl",
                                                                  "biocyc21_tier3_9392_triumpf_y.pkl",
                                                                  "biocyc21_tier3_9392_pp_leads_y.pkl"],
                        help="The y file name. (default value: ['biocyc21_tier3_9392_y.pkl', "
                             "'biocyc21_tier3_9392_minpath_y.pkl', 'biocyc21_tier3_9392_mllr_y.pkl', "
                             "'biocyc21_tier3_9392_triumpf_y.pkl', 'biocyc21_tier3_9392_pp_leads_y.pkl']).")
    parser.add_argument("--source-name", nargs="+", type=str,
                        default=['pathologic', 'minpath', 'mllr', 'triumpf', 'leads'],
                        help="The y source name. (default value: ['pathologic', 'minpath', 'mllr', 'triumpf', 'leads']).")
    parser.add_argument('--X-trust-name', type=str, default='biocyc21_tier12_41_Xe.pkl',
                        help='The X trusted file name. (default value: "biocyc21_tier12_41_Xe.pkl")')
    parser.add_argument('--y-trust-name', type=str, default='biocyc21_tier12_41_y.pkl',
                        help='The y trusted file name. (default value: "biocyc21_tier12_41_y.pkl")')
    parser.add_argument('--samples-ids', type=str, default='mltS_samples.pkl',
                        help='The samples ids file name. (default value: "mltS_samples.pkl")')
    parser.add_argument('--similarity-name', type=str, default='pathway_similarity_cos.pkl',
                        help='The labels similarity file name. (default value: "pathway_similarity_cos.pkl")')
    parser.add_argument('--vocab-name', type=str, default='vocab_biocyc.pkl',
                        help='The vocab file name. (default value: "vocab_biocyc.pkl")')
    parser.add_argument('--file-name', type=str, default='mltS',
                        help='The file name to save an object. (default value: "mltS")')
    parser.add_argument('--model-name', type=str, default='mltS',
                        help='The file name, excluding extension, to save an object. (default value: "mltS")')
    parser.add_argument('--dsname', type=str, default='golden',
                        help='The data name used for evaluation. (default value: "golden")')

    ###***************************     Preprocessing arguments      ***************************###
    parser.add_argument('--preprocess-dataset', action='store_true', default=False,
                        help='Preprocess biocyc collection.  (default value: False).')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='The dataset test size between 0.0 and 1.0. (default value: 0.2)')
    parser.add_argument('--binarize', action='store_true', default=False,
                        help='Whether binarize data (set feature values to 0 or 1). (default value: False).')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Whether to normalize data. (default value: False).')
    parser.add_argument('--use-external-features', action='store_true', default=False,
                        help='Whether to use external features that are included in data. '
                             '(default value: False).')
    parser.add_argument('--cutting-point', type=int, default=3650,
                        help='The cutting point after which binarize operation is halted in data. '
                             '(default value: 3650).')

    ###***************************        Training arguments        ***************************###
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train the mltS model. (default value: False).')
    parser.add_argument('--fit-intercept', action='store_false', default=True,
                        help='Whether the intercept should be estimated or not. (default value: True).')
    parser.add_argument('--train-selected-sample', action='store_true', default=False,
                        help='Train based on selected sample ids. (default value: False)')
    parser.add_argument('--ssample-input-size', type=float, default=0.7,
                        help='The size of input subsample. (default value: 0.7)')
    parser.add_argument('--ssample-mini-input-size', type=float, default=0.3,
                        help='The size of mini batch input data. (default value: 0.3)')
    parser.add_argument('--ssample-label-size', type=int, default=50,
                        help='Maximum number of labels to be sampled. (default value: 50).')
    parser.add_argument('--calc-subsample-size', type=int, default=50,
                        help='Compute loss over selected samples. (default value: 50).')
    parser.add_argument("--calc-label-cost", action='store_false', default=True,
                        help="Compute label cost, i.e., cost of labels. (default value: True).")
    parser.add_argument("--calc-total-cost", action='store_true', default=False,
                        help="Compute total cost, i.e., cost of bags plus cost of labels."
                             " (default value: False).")
    parser.add_argument('--label-bag-sim', action='store_true', default=False,
                        help='Whether to apply similarity constraint among labels within a bag. (default value: False).')
    parser.add_argument('--label-closeness-sim', action='store_true', default=False,
                        help='Whether to apply closeness constraint of a label to other labels of a bag. '
                             '(default value: False).')
    parser.add_argument('--corr-label-sim', action='store_true', default=False,
                        help='Whether to apply similarity constraint among labels. (default value: False).')
    parser.add_argument('--corr-input-sim', action='store_true', default=False,
                        help='Whether to apply similarity constraint among instances. (default value: False).')
    parser.add_argument("--early-stop", action='store_true', default=False,
                        help="Whether to terminate training based on relative change "
                             "between two consecutive iterations. (default value: False).")
    parser.add_argument("--loss-threshold", type=float, default=0.05,
                        help="A hyper-parameter for deciding the cutoff threshold of the differences "
                             "of loss between two consecutive rounds. (default value: 0.05).")
    # apply trusted reference datasets
    parser.add_argument('--use-trusted', action='store_true', default=False,
                        help='Train based on trusted data. (default value: False)')
    parser.add_argument('--beta-source', type=float, default=0.001,
                        help='Constant that applies for source specific weights. (default value: 0.001).')
    parser.add_argument('--apply-trusted-loss', action='store_true', default=False,
                        help='Apply discrepancy based on trusted source loss. (default value: False)')
    # apply peer learning
    parser.add_argument('--apply-peer', action='store_true', default=False,
                        help='Learn from peers. (default value: False)')
    parser.add_argument('--beta-peer', type=float, default=0.75,
                        help='Constant that scales the amount to absorb information from global '
                             'learner to individual learning. (default value: 0.75)')
    # partial labeling
    parser.add_argument('--apply-partial-label', action='store_true', default=False,
                        help='Fill the possible candidate labels for each example. (default value: False)')
    parser.add_argument('--use-trusted-partial-label', action='store_true', default=False,
                        help='Use trusted data for partial labeling. (default value: False)')
    parser.add_argument("--tau-partial", type=float, default=0.005,
                        help="The cutoff threshold for decision if partial label "
                             "learning is enables. (default value: 0.005)")
    parser.add_argument('--alpha-partial', type=float, default=0.75,
                        help='Constant that scales the labeling information inherited from '
                             'iterative propagation and the initial labeling confidence. '
                             '(default value: 0.75)')
    parser.add_argument('--num-neighbors', type=int, default=10,
                        help='Number of neighbours used for partial labeling. '
                             '(default value: 10)')
    parser.add_argument('--sigma', type=float, default=2,
                        help='Constant that scales the amount of Laplacian norm regularization '
                             'parameters. (default value: 2)')
    # apply regularization
    parser.add_argument('--alpha-glob', type=float, default=5,
                        help='Constant controlling trade-off between local and global weights. '
                             '(default value: 5)')
    parser.add_argument('--penalty', type=str, default='l21', choices=['l1', 'l2', 'elasticnet', 'l21'],
                        help='The penalty (aka regularization term) to be used. (default value: "l21")')
    parser.add_argument('--fuse-weight', action='store_true', default=False,
                        help='Whether to apply fused parameters technique. (default value: False).')
    parser.add_argument('--alpha-elastic', type=float, default=0.0001,
                        help='Constant that multiplies the regularization term to control '
                             'the amount to regularize parameters and in our paper it is lambda. '
                             '(default value: 0.0001)')
    parser.add_argument('--l2-ratio', type=float, default=0.35,
                        help='The elastic net mixing parameter, with 0 <= l2_ratio <= 1. l2_ratio=0 '
                             'corresponds to L1 penalty, l2_ratio=1 to L2. (default value: 0.35)')
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.01, 0.01, 0.01, 0.01, 10],
                        help="Five hyper-parameters for constraints. (default value: [0.01, 0.01, 0.01, 0.01, 10]).")
    parser.add_argument('--kappa', type=float, default=0.01,
                        help='Constant that scales the amount of S parameters. (default value: 0.01)')
    # apply learning hyperparameter
    parser.add_argument('--learning-type', type=str, default='optimal', choices=['optimal', 'sgd'],
                        help='The learning rate schedule. (default value: "optimal")')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='The learning rate. (default value: 0.0001).')
    parser.add_argument('--lr0', type=float, default=0.0,
                        help='The initial learning rate. (default value: 0.0).')
    parser.add_argument('--fr', type=float, default=0.9,
                        help='Forgetting rate to control how quickly old information is forgotten. The value should '
                             'be set between (0.5, 1.0] to guarantee asymptotic convergence. (default value: 0.7).')
    parser.add_argument('--delay', type=float, default=1.,
                        help='Delay factor down weights early iterations. (default value: 0.9).')

    ###***************************       Prediction arguments       ***************************###
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate mltS\'s performances. (default value: False).')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='Whether to predict bags_labels distribution from inputs using mltS. '
                             '(default value: False).')
    parser.add_argument('--meta-predict', action='store_true', default=False,
                        help='Whether to use global parameters for prediction. (default value: True).')
    parser.add_argument('--meta-adaptive', action='store_true', default=False,
                        help='Whether to predict labels based on the confidence table. (default value: False).')
    parser.add_argument('--meta-omega', action='store_true', default=False,
                        help='Whether to predict labels based on the source related weights. (default value: False).')
    parser.add_argument('--pref-model', type=str, default=None,
                        help='Whether to use a specific model\'s parameters for prediction. (default value: None).')
    parser.add_argument("--decision-threshold", type=float, default=0.5,
                        help="The cutoff threshold for mltS. (default value: 0.5)")
    parser.add_argument('--soft-voting', action='store_true', default=False,
                        help='Whether to predict labels based on the calibrated sums of the '
                             'predicted probabilities from an ensemble. (default value: False).')
    parser.add_argument('--pref-rank', action='store_true', default=False,
                        help='Whether to predict labels based on ranking strategy. (default value: False).')
    parser.add_argument('--top-k-rank', type=int, default=200,
                        help='Top k labels to be considered for predicting. Only considered when'
                             ' the prediction strategy is set to "pref-rank" option. (default value: 200).')
    parser.add_argument('--estimate-prob', action='store_true', default=False,
                        help='Whether to return prediction of labels and bags as probability '
                             'estimate or not. (default value: False).')
    parser.add_argument('--apply-tcriterion', action='store_true', default=False,
                        help='Whether to employ adaptive strategy during prediction. (default value: False).')
    parser.add_argument('--adaptive-beta', type=float, default=0.45,
                        help='The adaptive parameter for prediction. (default value: 0.45).')
    parser.add_argument('--pathway-report', action='store_true', default=False,
                        help='Whether to generate a detailed report for pathways for each instance. '
                             '(default value: False).')
    parser.add_argument('--no-parse', action='store_true', default=False,
                        help='Whether to parse Pathologic format file (pf) from a folder (default value: False).')
    parser.add_argument('--no-build-features', action='store_true', default=False,
                        help='Whether to construct features (default value: True).')
    parser.add_argument('--top-k-taxa', type=int, default=100,
                        help='Top k taxa to be considered for reporting. (default value: 100).')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Whether to produce various plots from predicted outputs. '
                             '(default value: False).')
    parser.add_argument('--psp-k', type=int, default=10,
                        help='K value for the propensity score. (default value: 10).')

    parse_args = parser.parse_args()
    args = __internal_args(parse_args)

    train(arg=args)


if __name__ == "__main__":
    parse_command_line()
