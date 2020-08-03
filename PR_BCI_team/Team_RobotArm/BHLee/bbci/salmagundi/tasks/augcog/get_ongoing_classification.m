function traces = get_ongoing_classification(blk,varargin);
%GET_ONGOING_CLASSIFICATION TRAINS A LEAVE_TWO_NEIGHBORED_BLOCK_OUT CLASSIFIER ON SOME SPECIFIC FEATURE AND APPLY IT TO THE CONTINUOUS TEST SIGNAL.
%
% usage: 
%      traces = get_ongoing_classification(file,<task='*auditory',feature='bandEnergy',ivallen=[5000 30000],step=100,classifier='LDA',...>);
%      traces = get_ongoing_classification(blk,<task='*auditory',feature='bandEnergy',ivallen=[5000 30000],step=100,classifier='LDA',...>);
%      traces = get_ongoing_classification(cnt,mrk,<feature='bandEnergy',ivallen=[5000 30000],step=100,classifier='LDA',...>);
% 
% input:
%    file       the name of an augcog file
%    feature    the used feature, for availability see get_augcog_* 
%    blk        a usual augcog blk structure
%    cnt,mrk    as usual
%    task       the used task
%    ivallen    first value: steps between to train events in msec.
%               second value: length of the window
%    step       the step width during apply, and optional as second value the window length (=ivallen(2))
%    classifier a classifier
%    ....       all other values are directly sent to get_augcog_feature
%
% output:
%    traces     a struct with following entries:
%             .x a matrix with number of LEAVE_ONE_BLOCK_OUT classifiers rows and classification results per time in the rows.
%             .y a matrix with number of LEAVE_ONE_BLOCK_OUT classifiers rows with 2 if the used data belongs to the train set of this classifier, 1 if some part belong to the train set, 0 otherwise.
%             .fs the sampling rate
%
% Guido Dornhege, 04/05/2004
% load the file, if necessary

global AUGCOG_VALIDATION
if isempty(AUGCOG_VALIDATION)
  AUGCOG_VALIDATION = 'LTNO';
end

% get features
if ~isstruct(blk)
  blk = getAugCogBlocks(blk);
end


% set defaults
opt = setdefault(varargin,{'task','feature','ivallen','step','classifier'},...
                          {'*auditory','bandPowerbyvariance',[5000 30000],100,'LDA'});

varargin = varargin(6:end);

if length(opt.ivallen) == 1
  opt.ivallen = [1,1]*opt.ivallen;
end

if length(opt.step) == 1
  opt.step = [opt.step,opt.ivallen(2)];
end

% Choose the task
if ~isempty(opt.task) & ~isstruct(opt.task)
  if ~iscell(opt.task), opt.task = {opt.task};end
  blk = blk_selectBlocks(blk,opt.task{:});
end

% read the blocks
if ~isstruct(opt.task)
  [cnt,mrk] = readBlocks(blk,[],1);
else
  cnt = blk;
  mrk = opt.task;
end

% feature extraction
[fv,apply] = feval(['get_augcog_' opt.feature],cnt,mrk,opt.ivallen,varargin{:});
mr = separate_markers(mrk);
%CLassification
bi = unique(fv.bidx);
ivis = [mr.ival(1,1),mr.ival(2,1:end-1)+1;mr.ival(2,:)];
switch AUGCOG_VALIDATION
 case 'LTNO'
  clear divTe divTr;                   
  for i = 1:length(bi)-1;
    divTe{i} = {bi([i,i+1])};
    divTr{i} = {bi([1:i-1,i+2:length(bi)])};
    if i==1
      in = find(sum(mr.y(1:2,:),1));
    else
      in = find(mr.y(i+1,:));
    end
    
    Ar{i} = ivis(:,in)'*1000/mrk.fs;
  end
 case 'LOO'
  clear divTe divTr;
  for i = 1:length(bi);
    divTe{i} = {bi(i)};
    divTr{i} = {bi([1:i-1,i+1:length(bi)])};
    in = find(mr.y(i,:));
    Ar{i} = ivis(:,in)'*1000/mrk.fs;
  end
 case 'LRO'
  clear divTe divTr;
  for i = 1:2:length(bi)
    j = 0.5*(i+1);
    if i==length(bi)
      divTe{j} = {bi(i)};
      in = find(mr.y(i,:));
    else
      divTe{j} = {[bi(i),bi(i+1)]};
      in = find(sum(mr.y([i,i+1],:),1));
    end
    divTr{j} = {[bi(1:i-1),bi(i+2:end)]};
    Ar{j} = ivis(:,in)'*1000/mrk.fs;
  end
end

fv.divTr = divTr;fv.divTe = divTe;
C = xvalidation(fv,opt.classifier,struct('xTrials',[1 1],'msTrials',[1 1],'train_only',1));
% relevant area:
% $$$ switch 1
% $$$  case 1
% $$$ Ar{1} = [mrk.pos(1),mrk.pos(3)-1]*1000/mrk.fs;
% $$$ for j = 2:length(fv.taskname)-2;
% $$$   Ar{j} = [mrk.pos(j+1),mrk.pos(j+2)-1]*1000/mrk.fs;
% $$$ end
% $$$ Ar{length(fv.taskname)-1} = [mrk.pos(end),mrk.end]*1000/mrk.fs;
% $$$ case 2
% $$$ Ar{1} = [mrk.pos(1),mrk.pos(2)-1]*1000/mrk.fs;
% $$$ for j = 2:length(fv.taskname)-2;
% $$$   Ar{j} = [mrk.pos(j),mrk.pos(j+1)-1]*1000/mrk.fs;
% $$$ end
% $$$ Ar{length(fv.taskname)-1} = [mrk.pos(end-1),mrk.end]*1000/mrk.fs;
% $$$ end

cla = opt.classifier;
if isstruct(cla)
  cla = cla.classy;
elseif iscell(cla)
  cla = cla{1};
end

traces = get_ongoing(cnt,mrk,opt.step,cla,C,Ar,apply);
traces.className = fv.className;



