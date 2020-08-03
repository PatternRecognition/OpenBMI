prefix= 'result_tables/alternative_classification/results_cls_';

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

opt= strukt('dataset_names', subdir_list);
%opt= strukt('dataset_names', subdir_list, ...
%            'row_summary', 'median');

%meths= {'csp_orig', 'csp_auto', 'laplace'};
meths= {'csp_orig', 'csp_auto', 'csp_auto_filtsel*', 'laplace'};
latex_results('alternative_classification', [prefix 'baseline'], ...
              opt, 'methods', meths);
