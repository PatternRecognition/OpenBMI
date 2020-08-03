% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt, analyze.features.clab));
cont_proc = struct('clab',{clab});

if isfield(bbci, 'marker_output'),
  marker_output= bbci.marker_output;
else
  marker_output= struct();
end
marker_output= set_defaults(marker_output, ...
                            'marker', num2cell([11:49]), ...
                            'value', [1:10, 1:10, 1:10, 1:10], ...
                            'no_marker', 0);

os= 1000/bbci.fs;  % duration of one sample in msec

feature = struct('cnt',1);
feature.ilen_apply = analyze.ival(end) - analyze.ref_ival(1) + os; 
% + one sample to account for difference in marker position interpretation

feature.proc= {'proc_baseline', ...
               'proc_jumpingMeans'};
feature.proc_param= {{diff(analyze.ref_ival), 'beginning_exact'}, ...
                     {analyze.ival-analyze.ival(end)}};

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features, opt.model);
str1 = sprintf('%g,',[bbci.classDef{1,:}]);
cls.condition = sprintf('M({{%s},[%g %g]});',str1(1:end-1), analyze.ival(end)*[1 1] - os);

% bbci.cont_proc= cont_proc;
% bbci.feature= feature;
% bbci.marker_output= marker_output;
% bbci.classifier= cls;