prefix= 'result_tables/alternative_classification/results_cls_';

subdir_list_naive= textread([BCI_DIR 'studies/season2/session_list'], '%s');
subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');
ext= setdiff(subdir_list, subdir_list_naive);
skip= strpatternmatch(ext, subdir_list);
skip= [skip strpatternmatch('VPco*', subdir_list)];

%opt= strukt('dataset_names', subdir_list);
opt= strukt('dataset_names', subdir_list, ...
            'skip_rows', skip, ...
            'row_summary', 'median');

meths= {'csp_orig_redo_csp', 'lapcsp_orig'};
%meths= {'csp_orig', 'lapcsp_orig', 'csp_auto', 'lapcsp_auto'};
%meths= {'csp_orig', 'csprda_orig'};
latex_results('alternative_classification', [prefix 'lapcsp_vs_csp'], ...
              opt, 'methods', meths);
