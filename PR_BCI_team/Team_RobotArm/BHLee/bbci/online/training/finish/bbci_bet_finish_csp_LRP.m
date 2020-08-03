% a combination of cspauto and LRP
%
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls
% xx{1} stands for csp, xx{2} stands for LRP, xx{3} stands for the
% additional LRP dummy-classifier that manges the offset!
%
% Johannes 07/2011

os= 1000/bbci.fs;  % duration of one sample in msec


% Extract channel labels clab for each classifier
cont_proc = struct('clab',{...
    Cnt.clab(chanind(Cnt, analyze{1}.clab));
    Cnt.clab(chanind(Cnt, analyze{2}.clab))}, ...
    'procFunc', {{'online_linearDerivation','online_filt'};
                {'online_filt'}},...
    'procParam', { {{analyze{1}.csp_w} ,{analyze{1}.csp_b,analyze{1}.csp_a} };
                 {{analyze{2}.filt_b,analyze{2}.filt_a}}} );

fusion_param = {strukt('num_cl_out', 3, ...
    'ixRawOutNeglect', 3, ...
    'action', {'do_nothing', 'do_nothing', 'sum'}, ...
    'action_ixInput', { [] [] [1 2]}) };

post_proc = strukt('proc', 'bbci_bet_post_proc_fusion', ...
    'proc_param', fusion_param);   

%extract the marker corresponding to the classes (foot-foot). bbci.classDef
% has all infos, bbci.classes just the labels to use...
dum = cell2mat(bbci.classDef(1,[find(strcmp(bbci.classDef(2,:), bbci.classes(1))) find(strcmp(bbci.classDef(2,:), bbci.classes(2)))] ));

marker_output= struct( ...
    'marker', {dum; dum}, ... % aus bbci_bet_analyze
    'value', {dum; dum}, ...
    'no_marker', {0; 0});





if ~iscell(opt.model) & ischar(opt.model)
    opt.model = {opt.model; opt.model};
    warning('since only specified for one classifier, opt.model of csp and LRP were set to %s', opt.model{1});
end




cls = struct('fv',{1;2;3}, ...
             'applyFcn', {getApplyFuncName(opt.model{1}); 
                          getApplyFuncName(opt.model{2}); 
                          'apply_nothing'}, ...
             'C', {trainClassifier(analyze{1}.features, opt.model{1});
                   trainClassifier(analyze{2}.features, opt.model{2});
                   []} ...
               );

%           cls(1).C.mean = []; %evoke error in adaptation!!
           
%add the baselining condition to cls{3}
str1 = sprintf('%g,',[bbci.classDef{1,:}]);
cls(3).condition= sprintf('M({{%s},[%g %g]});',str1(1:end-1), analyze{2}.baseline(end)*[1 1] - os);
                       
if isnumeric(opt.ilen_apply) & length(opt.ilen_apply) == 1
    opt.ilen_apply = {opt.ilen_apply; opt.ilen_apply};
    warning('since only specified for one classifier, ilen_apply of csp and LRP were set to %i', opt.ilen_apply{1});
end

feature = struct('cnt',{1;2;2}, ...
                 'ilen_apply', {opt.ilen_apply{1}; %opt.ilen_apply has to be specified for csp and LRP!!!
                                opt.ilen_apply{2}; ...
                                diff(analyze{2}.baseline) + os}, ...
                 'proc', {{'proc_variance','proc_logarithm'};
                          {'proc_meanAcrossTime','proc_subtractFVREF'}; ...
                          {'proc_meanAcrossTime','proc_saveFVREF'} }, ...
                 'proc_param', { {{},{}}; ...  %no params for processing functions
                                 {{},{}}; ...
                                 {{},{}} });
                       