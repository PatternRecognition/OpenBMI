prefix= 'result_tables/alternative_classification/results_cls_';

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

opt= strukt('dataset_names', subdir_list, ...
            'skip_rows', 3, ...
            'header', 'portrait');

meths= {'*'};

latex_results('alternative_classification/temp', [prefix 'outl'], ...
              opt, 'methods', meths);
