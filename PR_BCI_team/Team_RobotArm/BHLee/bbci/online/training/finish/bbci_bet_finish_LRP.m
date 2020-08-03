% Prepare the information needed for an LRP online experiment
% Compared to ERD experiments, for LRP an additional baseline correction vector has to be 
% preserved from the start of each trial to the sliding windows within a trial 
% (for incremental online feedback)
%
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls
%
%
%
% see also: proc_saveFVREF.m, proc_subtractFVREF.m
%
% 01/2011 by David, Michael, Benjamin


os= 1000/bbci.fs;  % duration of one sample in msec

% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt, analyze.clab));
cont_proc = struct('clab',{clab});


if isfield(bbci, 'marker_output'),
  marker_output= bbci.marker_output;
else
  marker_output= struct();
end
marker_output= set_defaults(marker_output, ...
                            'marker', bbci.classDef(1,:), ... % aus bbci_bet_analyze
                            'value', [1:2], ...
                            'no_marker', 0);
% For main classifier
feature= struct('cnt',1);
feature.ilen_apply = opt.ilen_apply;
feature.proc = {'proc_meanAcrossTime','proc_subtractFVREF'};
feature.proc_param = {{},{}};

% eventuell nicht nötig, noch Mittelwerte abzuziehen und zu normalisieren...
%feature.proc = {'proc_meanAcrossTime','proc_subtractFVREF','proc_flaten','proc_subtractMean', 'proc_normalize'};
%feature.proc_param = {{},{},{struct('force_flaten', 1)},{opt.meanOpt},{opt.normOpt}};

% For baseline correction
feature(2).cnt= 1;
feature(2).ilen_apply= diff(analyze.baseline) + os;
feature(2).proc= {'proc_meanAcrossTime','proc_saveFVREF'};
feature(2).proc_param= {{},{}};

% For main classifier
cls= struct('fv',1);
cls.applyFcn= getApplyFuncName(opt.model);
cls.C= trainClassifier(analyze.features,opt.model);

% For baseline correction
cls(2).fv= 2;
cls(2).applyFcn= 'apply_nothing';
cls(2).C= [];
str1 = sprintf('%g,',[bbci.classDef{1,:}]);
cls(2).condition= sprintf('M({{%s},[%g %g]});',str1(1:end-1), analyze.baseline(end)*[1 1] - os);
