
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt,opt.clab));
cont_proc = struct('clab',{clab});
% $$$ if bbci.player==2
% $$$   for i = 1:length(cont_proc.clab)
% $$$     cont_proc.clab{i} = ['x',cont_proc.clab{i}];
% $$$   end
% $$$ end
cont_proc.procFunc = {};
cont_proc.procParam = {};

feature = struct('cnt',1);
feature.ilen_apply = diff(opt.ival);
if isfield(opt,'baseline') & ~isempty(opt.baseline)
  feature.proc = {'proc_baseline','proc_selectIval','proc_jumpingMeans'};
  feature.proc_param = {{opt.baseline},{opt.selectival},{opt.jMeans}};
else
  feature.proc = {'proc_jumpingMeans'};
  feature.proc_param = {{opt.jMeans}};
end

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features,opt.model);
str1 = sprintf('%g,',[bbci.classDef{1,:}]);
cls.condition = sprintf('M({{%s},[%g %g]});',str1(1:end-1),opt.ival(end)*[1,1]);
bbci.errorJit = opt.ival(end);