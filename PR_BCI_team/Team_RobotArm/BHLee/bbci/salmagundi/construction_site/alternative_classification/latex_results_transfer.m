prefix= 'result_tables/alternative_classification/results_trf_';

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');
bad_perf= {'VPcn*','VPco*','VPcq*','Friederike_06_02_21'};
skip= strpatternmatch(bad_perf, subdir_list);

opt= strukt('dataset_names', subdir_list, ...
            'skip_rows', skip, ...
            'header', 'landscape');

% meths= {'csp_*', 'laplace' 'logistic_rank2' 'specCSP','GP*'};
meths = {'*'};

latex_results('alternative_transfer', [prefix 'competition'], ...
              opt, 'methods', meths);
