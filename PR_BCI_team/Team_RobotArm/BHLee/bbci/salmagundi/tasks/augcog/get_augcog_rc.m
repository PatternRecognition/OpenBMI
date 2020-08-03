function [fv,apply] = get_augcog_rc(blk,varargin);
%GET_AUGCOG_RC OPENS AN AUGCOG FILE AND CALCULATES RC COEFFICIENTS
%
% usage:
%     fv = get_augcog_rc(file,<task=all available,ivallen=1000,band=[7 30],spatial='diagonal',ar=6,rc=ar,,channels={'not','E*','M*'});
%     fv = get_augcog_rc(blk,<task=all available,ivallen=1000,band=[7 30],spatial='diagonal',ar=6,rc=ar,channels={'not','E*','M*'});
%     fv = get_augcog_rc(cnt,mrk,<ivallen=1000,band=[7 30],spatial='diagonal',ar=6,rc=ar,channels={'not','E*','M*'});
%
% input:
%     file     -  the name of an augcog file
%     blk      -  a usual augcog blk structure
%     cnt,mrk  -  as usual
%     task     -  the task of the subject, e.g. '*auditory' 
%                 (for more than one as a cell array)
%     ivallen  -  the length of the ival (to make epochs)
%     band     -  the band spectral information should be given back, can have more than one row.
%     spatial  -  the spatial filter to use see proc_spatial_filtering_augcog for documentation
%     ar       - the number of ar coefficients
%     rc       - the number of rc coefficients
%
%
% output:
%     fv       - the resulting feature vector as a struct with fields:
%                .x    a spec*channel*trials matrix
%                .y    the logical 2xtrials matrix
%                .className  = {'low','high'};
%                .t    the time bins
%                .clab a cell array consisting of the channel names
%                .task a logical array to divide into different sessions
%                .taskname a cell array of task names concerning .task
%
% Guido Dornhege, 27/04/2004

% load the file, if necessary
if ~isstruct(blk)
  blk = getAugCogBlocks(blk);
end

% set defaults
opt = setdefault(varargin,{'task','ivallen','band','spatial','ar','rc','channels'},...
                          {[],1000,[3 40],'diagonal',6,[],{'not','E*','M*'}});

if isempty(opt.rc), opt.rc = opt.ar;end

if length(opt.ivallen) == 1
  opt.ivallen = [1,1]*opt.ivallen;
end


% Choose the task
if ~isempty(opt.task) & ~isstruct(opt.task)
  if ~iscell(opt.task), opt.task = {opt.task};end
  blk = blk_selectBlocks(blk,opt.task{:});
end

% read the blocks
if ~isstruct(opt.task)
  [cnt,mrk] = readBlocks(blk);
else
  cnt = blk;
  mrk = opt.task;
end

% choose channels
if ~isempty(opt.channels)
  apply.chans = chanind(cnt,opt.channels{:});
  cnt = proc_selectChannels(cnt,opt.channels{:});
end

nchan = size(cnt.x,2);
% band pass filtering
par = {};
for i = 1:size(opt.band,1)
  [b,a] = butter(5,opt.band(i,:)/cnt.fs*2);
  par = {par{:},b,a};
  cc = proc_filt(cnt,b,a);
  if i ==1
    ccc = cc;
  else
    ccc.x = cat(2,ccc.x,cc.x);
  end
end
if size(opt.band,1)>0
  apply.cnt_apply = struct('fcn','proc_filtconcat','param',{par});
else
  apply.cnt_apply = [];
end

cnt = ccc;

% spatial filtering
[h,apply.cnt_apply(length(apply.cnt_apply)+1)] = proc_spatial_filtering_augcog(cnt,opt.spatial);
if isstruct(h);
  cnt = h;
end


% epoching
mrk = separate_markers(mrk);
mrk = mrk_setMarkers(mrk,opt.ivallen);

epo = makeEpochs(cnt,mrk,[0 opt.ivallen(2)]);

clear cnt mrk blk
make_proc_apply('init')
% maybe spatialling again???
if isnumeric(h)
  [epo,proc_apply] = proc_spatial_filtering_augcog(epo,opt.spatial);
  make_proc_apply(proc_apply);
end


epo.task = epo.y; epo.taskname = epo.className;
epo.indexedByEpochs = {'task','bidx'};
clInd0 = getClassIndices(epo,'base*');
clInd = getClassIndices(epo,'low*');
clInd2 = getClassIndices(epo,'high*');

epo.y = zeros(3,size(epo.y,2));
epo.className = {'base','low','high'};

ind0 = find(sum(epo.task(clInd0,:),1));
ind = find(sum(epo.task(clInd,:),1));
ind2 = find(sum(epo.task(clInd2,:),1));

epo.y(1,ind0) = 1;
epo.y(2,ind) = 1;
epo.y(3,ind2) = 1;

indi = find(sum(epo.y,2)>0);
epo.y = epo.y(indi,:);
epo.className = epo.className(indi);

epo.bidx = (1:size(epo.task,1))*epo.task;

% rc
fv = proc_rcCoefsPlusVar(epo,opt.ar,opt.rc);


make_proc_apply(sprintf('fv=proc_rcCoefsPlusVar(epo,%i,%i);',opt.ar,opt.rc));


apply.proc_apply = make_proc_apply('get');
