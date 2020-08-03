% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

os= 1000/bbci.fs;  % duration of one sample in msec

%% Set Variables for ERP Speller
% Extract channel labels clab
cont_proc = struct('clab',{Cnt.clab(chanind(Cnt, analyze{1}.features.clab));
                           Cnt.clab(chanind(Cnt, analyze{2}.features.clab))});

marker_output = struct('no_marker', {0;0}, ...
                       'marker', {num2cell([11:16 21:26 31:36 41:46]);
                                  num2cell([51:56 61:66 151:156 161:166])}, ...
                       'value', {[1:6, 1:6, 1:6, 1:6];
                                 [7:12, 7:12, 7:12, 7:12]});

feature = struct('cnt',{1;2}, ...
                 'ilen_apply', {analyze{1}.ival(end) - analyze{1}.ref_ival(1) + os;
                                analyze{2}.ival(end) - analyze{2}.ref_ival(1) + os}, ...
                 'proc', {{'proc_baseline', 'proc_jumpingMeans'};
                          {'proc_baseline', 'proc_jumpingMeans'}}, ...
                 'proc_param', {{{diff(analyze{1}.ref_ival), 'beginning_exact'}, ...
                                 {analyze{1}.ival-analyze{1}.ival(end)}};
                                {{diff(analyze{2}.ref_ival), 'beginning_exact'}, ...
                                 {analyze{2}.ival-analyze{2}.ival(end)}}});

str1 = sprintf('%g,',[bbci.classDef{1,[1 2]}]);
str2 = sprintf('%g,',[bbci.classDef{1,3}]);
cls = struct('fv',{1;2}, ...
             'applyFcn', {getApplyFuncName(opt.model);
                          getApplyFuncName(opt.model)}, ...
             'C', {trainClassifier(analyze{1}.features, opt.model);
                   trainClassifier(analyze{2}.features, opt.model)}, ...
             'condition', {sprintf('M({{%s},[%g %g]});',str1(1:end-1), analyze{1}.ival(end)*[1 1] - os);
                           sprintf('M({{%s},[%g %g]});',str2(1:end-1), analyze{2}.ival(end)*[1 1] - os)});


%% Adjust ErrP Classifier Bias:
cls(2).C.b = cls(2).C.b - bbci.setup_opts.ErrP_bias;
