
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
cont_proc.proc = {};
cont_proc.proc_param = {};

% Features: laplace filter
% again:channels selecting.

feature = struct('cnt',1);
feature.ilen_apply = opt.ilen_apply;
feature.proc = {};
feature.proc_param = {};
% laplace filtering
if opt.laplace
  feature.proc{end+1} = 'proc_laplace';
  feature.proc_param{end+1} = {};
end
% selecting channels
feature.proc{end+1} = 'proc_selectChannels';
feature.proc_param{end+1} = {opt.classiclab{:}};
% applying the FFT filter
feature.proc{end+1} = 'proc_filtBruteFFT';
feature.proc_param{end+1} = {opt.band,opt.fftparams{:}};
% jumping Means filtering
if isfield(opt,'jMeans')&~isempty(opt.jMeans)
  feature.proc{end+1} = 'proc_jumpingMeans';
  feature.proc_param{end+1} = {opt.jMeans};
end

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features,opt.model);

