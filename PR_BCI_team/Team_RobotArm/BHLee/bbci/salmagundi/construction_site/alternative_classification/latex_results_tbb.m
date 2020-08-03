prefix= 'result_tables/alternative_classification/results_trf_';

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

opt= strukt('dataset_names', subdir_list);

meths= {'csp_orig', 'csp_auto', 'csp_auto_filtsel', 'csp_auto_rej*', 'csp_auto_full', 'laplace'};

latex_results('alternative_transfer', [prefix 'baseline'], ...
              opt, 'methods', meths);
