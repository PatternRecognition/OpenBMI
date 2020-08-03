function [testTube, outTraces, labels]= ...
    plot_tube_detection(val, method, dsply, appendix)
%[testTube, traces]= plot_tube_detection(val, method, dsply, <appendix>)
%
% warning: do not work correctly with x-validation so far!

% train.idx: [] for all, [0-1] fraction for cross-validation, or vector of idcs

if ~iscell(val.train_file), val.train_file= {val.train_file}; end
if ~isfield(val, 'test_file'), val.test_file= val.train_file; end
if ~iscell(val.test_file), val.test_file= {val.train_file}; end

if ~isfield(method,'jit'), method.jit=0; end
if ~isfield(method,'separateActionClasses'),  
  method.separateActionClasses= 0; 
end
if ~isfield(method,'combinerFcn'), 
  method.combinerFcn= inline('x');
end

if ~isstruct(dsply), 
  dsply= struct('E',dsply);
end
if ~isfield(dsply, 'scale'), dsply.scale= 1; end
if ~isfield(dsply, 'tubePercent'), dsply.tubePercent=[5 10 15]; end
if ~isfield(dsply, 'facealpha'), dsply.facealpha=1; end

[cnt, mrk, mnt, N]= concatProcessedEEG(cat(2, val.train_file, val.test_file));
%% prevent from using EMG, EOG channels
cnt= proc_selectChannels(cnt, 'not','E*');

nEvents= sum(N);
if isempty(val.xTrials),
  nTrains= sum(N(1:length(val.train_file)));
  if isempty(val.train_idx),
    train_idx= 1:nTrains;
  elseif length(val.train_idx)==1,
    nTrains= round(val.train_idx*nTrains);
    train_idx= 1:nTrains;
  end
  if isempty(val.test_idx),
    test_idx= nTrains+1:nEvents;
  end
  cnt.divTr{1}= {train_idx};
  cnt.divTe{1}= {test_idx};
else
  warning('do not work correctly with x-validation so far!');
  [cnt.divTr, cnt.divTe]= sampleDivisions(mrk.y, val.xTrials);
  train_idx= 1:nEvents;
  test_idx= 1:nEvents;
end
if isfield(val, 'pace'),
  pairs= getEventPairs(mrk, val.pace);
  train_pairs= metaintersect(pairs, train_idx);
  equi_set= equiSubset(train_pairs);
  equi_set= [equi_set{:}];
  for it= 1:length(cnt.divTr),
    cnt.divTr{it}= metaintersect(cnt.divTr{it}, equi_set);
  end
end
if isfield(method, 'chans'),
  cnt= proc_selectChannels(cnt, method.chans);
end
if isfield(method, 'proc_cnt'),
  eval(method.proc_cnt)
end

mrk_train= pickEvents(mrk, train_idx);
epo= makeEpochs(cnt, mrk_train, [-method.ilen 0], method.jit);
if ~method.separateActionClasses,
  epo.y= ones(1, size(epo.y,2));
  epo.className= {'action'};
end
no_moto= makeEpochs(cnt, mrk_train, [-method.ilen 0], method.jit_noevent);
no_moto.y= ones(1,size(no_moto.y,2));
no_moto.className= {'no event'};
epo= proc_appendEpochs(epo, no_moto);
clear no_moto
eval(method.proc);

if isfield(method,'msTrials'),
  fv2= copyStruct(fv, 'divTr','divTe');
%  if isfield(val, 'pace'),
%    fv2.equi= ...
%  end
  classy= selectModel(fv2, method.model, method.msTrials, 0);
  C= doXvalidationPlus(fv, classy, [], 'Train only');
else
  C= doXvalidationPlus(fv, method.model, [], 'Train only');
end
[func, params]= getFuncParam(method.model);

nShifts= length(dsply.E);
nTrials= length(epo.divTe);
nTestClasses= 1;
outTraces= zeros(nTrials, nShifts, length(test_idx));
mrk_test= pickEvents(mrk, test_idx);
test_cl= {1:size(mrk_test.y,2)};
tic;
for is= 1:nShifts, 
  epo= makeEpochs(cnt, mrk_test, dsply.E(is)+[-method.ilen 0]);
  eval(method.proc);
  fv= proc_flaten(fv);
  for it= 1:nTrials,
    nDiv= length(epo.divTe{it});
    for id= 1:nDiv,
      n= id+(it-1)*nDiv;
      idxTe= epo.divTe{it}{id};
      idxTe= find(ismember(test_idx, idxTe));
      out= applyClassifier(fv, method.model, C(n), idxTe);
      outTraces(it, is, idxTe)= dsply.scale * method.combinerFcn( out );
    end
  end
  for ic= 1:nTestClasses,
    outCl= outTraces(:, is, test_cl{ic});
    testTube(is, :, ic)= fractileValues(outCl(:), dsply.tubePercent);
  end
  print_progress(is, nShifts);
end

clf;
if dsply.facealpha==1,
  plotTube(testTube, dsply.E);
else
  plotTubeNoFaceAlpha(testTube, dsply.E);
end
if nTestClasses>1,
  legend(epo.className);
end
title(sprintf('%s  [%s]', untex(cnt.title), ...
              vec2str(dsply.tubePercent, '%d', ' / ')));

labels= mrk.y(:,test_idx);

if exist('appendix', 'var'),
  save(['traces_dtct_' appendix], ...
        'outTraces', 'labels', 'testTube', 'val', 'method', 'dsply');
end
