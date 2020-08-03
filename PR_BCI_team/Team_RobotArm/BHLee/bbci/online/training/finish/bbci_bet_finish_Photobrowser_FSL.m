
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
% clab = Cnt.clab(chanind(Cnt,opt.clab));
% Cnt = proc_selectChannels(Cnt, 'not', 'E*');
cont_proc = struct('clab',{epo.clab});

cont_proc.procFunc = {'online_filt'};
cont_proc.procParam = {{hpfilt.b,hpfilt.a}};
  
marker_output = struct();
marker_output.marker = {20};
marker_output.value = [20];
% marker_output.marker = {20:40};
% marker_output.value = [20:40];
marker_output.no_marker = 0;

feature = struct('cnt',1);
feature.ilen_apply = diff(opt.ival)+10; %10 to account for difference in marker position interpretation

feature.proc = {'proc_jumpingMeans','proc_flaten','proc_subtractMean', 'proc_normalize'};
feature.proc_param = {{opt.selectival-opt.ival(2)},{struct('force_flaten', 1)},{opt.meanOpt},{opt.normOpt}}; % -800 

% feature.proc = {'proc_jumpingMeans'};
% feature.proc_param = {{opt.selectival-opt.ival(2)-10}}; % -800 

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features,opt.model);
str1 = sprintf('%g,',[marker_output.marker{:}]);
%% WHY INTERVAL IN REVERSE ORDER?
cls.condition = sprintf('M({{%s},[%g %g]});',str1(1:end-1), opt.ival(end)*[1 1]);
% bbci.errorJit = opt.ival(end);