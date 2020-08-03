function traces = get_ongoing(blk,task,ivallen,classifier,C,Range,apply);
%GET_ONGOING_CLASSIFICATION APPLY A CLASSIFIER TO THE CONTINUOUS TEST SIGNAL.
%
% usage: 
%   traces = get_ongoing(cnt,mrk,ivallen,classifier,C,Range,apply);
%   traces = get_ongoing(file,task,ivallen,classifier,C,Range,apply);
%   traces = get_ongoing(blk,task,ivallen,classifier,C,Range,apply);
% input:
%    file       the name of an augcog file
%    feature    the used feature, for availability see get_augcog_* 
%    blk        a usual augcog blk structure
%    cnt,mrk    as usual
%    task       the used task
%    ivallen    first value: steps between to train events in msec.
%               second value: length of the window
%    classifier the name of the classifier (needed for constructing
%               the apply function
%    C          an array of trained  classifiers
%    Range         ranges in msec which classifier should work for
%    apply      structure given back by get_augcog_*, necessary to
%               extract the relevant features from the continuous data
%
% output:
% traces a struct with following entries: 
%              .x with the classifier
%             outputs regarding Range (NaN if not determinable) 
%              .y
%             with labels between -1 and 1, where for overlapping
%             trials the values could really between them (in the
%             ratio the window overlap in each class)
%             .fs the sampling rate
%             .t  endpoints in time regarding traces.x in msec
%
% Guido Dornhege, 04/05/2004
%
% Some more info by Mikio Braun,
%
% In conclusion, (see below), apply needs so much information that
% it only makes sense to call this function from within
% get_ongoing_classification. 
%
% Fields needed in apply:
%
%   In general, apply is set by get_augcog_* and thus contains the
%   relevant information about how the features were extracted.
%
%   apply.chans       indices of channels used
%   apply.cnt_apply   some general preprocessing of cnt. Spatial
%                     preprocessing is stored here
%   apply.proc_apply  a string containing code to map epo to fv and
%                     perform the feature extraction

%%------------------------------------------------------------
%% Parameter setup - the usual stuff
%%

% load the file, if necessary
if ~isstruct(blk)
  blk = getAugCogBlocks(blk);
end

% Choose the task
if ~isempty(task) & ~isstruct(task)
  if ~iscell(task), task = {task};end
  blk = blk_selectBlocks(blk,task{:});
end

% read the blocks
if ~isstruct(task)
  [cnt,mrk] = readBlocks(blk,[],1);
else
  cnt = blk;
  mrk = task;
end

% If range is empty, only one classifier should be passed. Then,
% this classifier is used for the whole range.
if ~exist('Range','var') | isempty(Range)
  Range = [mrk.pos(1),mrk.end]*1000/mrk.fs;
end

if ~iscell(Range)
  Range = {Range};
end

if length(Range)~=length(C)
  error('Number of classifiers and number of ranges does not match');
end

%%------------------------------------------------------------
%% Prepare Data
%%

% the APPLY parameter is supposed to contain information on how the
% features were extracted to train the classifiers.

% use apply.chans to extract the relevant channels
%  cf. processing/proc_selectChannels
cnt.x= cnt.x(:,apply.chans);
cnt.clab= {cnt.clab{apply.chans}};

% use apply.cnt_apply to preprocess the cnt data
if isfield(apply,'cnt_apply') & ~isempty(apply.cnt_apply) & isfield(apply.cnt_apply,'fcn')
  for i = 1:length(apply.cnt_apply)
    if ~isfield(apply.cnt_apply(i),'param') | isempty(apply.cnt_apply(i).param)
      cnt = feval(apply.cnt_apply(i).fcn,cnt);
    else
      cnt = feval(apply.cnt_apply(i).fcn,cnt,apply.cnt_apply(i).param{:});
    end
  end
end

% construct the basic traces structure.
% this includes the time indices in traces.t !
traces = struct('fs',1000/ivallen(1));

traces.t = ivallen(1):ivallen(1):(mrk.end*1000/mrk.fs);

traces.x = zeros(length(traces.t),1);
traces.y = zeros(length(traces.t),1);

epo = copyStruct(cnt,'x');

% construct a table which tells us which classifier is used when.
% This constructs a length(Range)*3 table (although this might not
% be obvious).
%
% The structure is as follows:
%   Col 1: index of classifier
%   Col 2: start index,
%   Col 3: end index.
%
Ran = [];
for i = 1:length(Range)
  % construct a column vector which has the same number of rows as
  % the Range{i}
  IND = i*ones(size(Range{i},1),1);
  Ran = [Ran; IND, Range{i}];
end

% determine which function is the classification function 
%   Mikio suggests: put this into a function of it own or use the
%   applyfunc field and function handles.
apply_fcn = ['apply_' classifier];
if ~exist('apply_fcn','file');
  % that this is the default is completetly non-obvious from code
  % in train_LDA, etc. 
  apply_fcn = 'apply_separatingHyperplane';
end


yyy = round(mrk.ival*1000/mrk.fs/ivallen(1));
for i = 1:size(yyy,2)
  traces.y(max(1,yyy(1,i)):min(yyy(2,i),length(traces.t))) = [-1 1]*mrk.y(:,i);
end
traces.y = movingAverage(traces.y,ivallen(2)/ivallen(1));
i = find(abs(traces.y)<eps*ivallen(2)/ivallen(1));
traces.y(i) = 0;
i = find(traces.y>1-eps*ivallen(2)/ivallen(1));
traces.y(i) = 1;
i = find(traces.y<-1+eps*ivallen(2)/ivallen(1));
traces.y(i) = -1;


% Well, I suppose here is where the action is happening :)
%  assumes that apply.proc_apply contains code (as string) to
%  correctly process a piece of code
for i = 1:length(traces.t);
  % some tourist information... (progress bar)
  fprintf('\r%2.1f    ',100*i/length(traces.t));
 
  % compute the indices relevant here
  % that is basically
  % [ traces.t(i) - ivallen(2), traces.t(i) ]
  enp = [traces.t(i) - ivallen(2), traces.t(i)];
  
  % determine which classifier is in charge for this piece of cnt data
  % (this has to be done here, because the structure of enp is
  % going to change in a few seconds)
  in = find(Ran(:,2)<=enp(2) & Ran(:,3)>enp(2));

  enp = round(enp*mrk.fs/1000);
  enp = enp(1):enp(2);

  % extracts the feature and applies the corresponding classifier
  if isempty(enp) | enp(1)<=0 | isempty(in)
    traces.x(i) = nan;
  else
    epo.x = cnt.x(enp,:);
    eval(apply.proc_apply);
    fv.x = fv.x(:);
    traces.x(i) = feval(apply_fcn,C(Ran(in,1)),fv.x);
  end    
  enp = enp(find(enp>0));
  
  % true label (together with transitions)
  
% $$$   traces.y(i) = mean([-1 1]*mrk.y(:,length(mrk.pos) - ...
% $$$ 				  sum((ones(length(mrk.pos),1)*enp) ...
% $$$ 				      < (mrk.pos'*ones(1,length(enp))))));
end

% if feature starts with a high block, swap the labels!!!
if strmatch('high',mrk.className{1})
  traces.y = -traces.y;
end

fprintf('\n');

