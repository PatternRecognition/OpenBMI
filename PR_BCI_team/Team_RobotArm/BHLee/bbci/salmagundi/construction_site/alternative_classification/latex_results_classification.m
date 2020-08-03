prefix= 'result_tables/alternative_classification/results_cls_';

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');
skip= strpatternmatch({'VPco*','VPcq*'}, subdir_list);

opt= strukt('dataset_names', subdir_list, ...
            'skip_rows', skip, ...
            'header', 'landscape');

%meths= {'csp_*', 'laplace*','spe*','GP*','fourier','aar','log*','arma*'};
meths= {'*'};

latex_results('alternative_classification', [prefix 'competition'], ...
              opt, 'methods', meths);
