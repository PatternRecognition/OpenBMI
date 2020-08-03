prefix= 'result_tables/alternative_classification/results_cls_';

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');
skip= strpatternmatch('VPco*', subdir_list);

%opt= strukt('dataset_names', subdir_list);
opt= strukt('dataset_names', subdir_list, ...
            'skip_rows', skip, ...
            'row_summary', 'median');

meths= {'csp_orig', 'csprda_orig'};
latex_results('alternative_classification', [prefix 'csprda_vs_csp'], ...
              opt, 'methods', meths);
