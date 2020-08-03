function epo= makeSegments(cnt, mrk, ival, jitter)
%epo= makeSegments(cnt, mrk, ival, <jitter>)
%
% IN   cnt    - continuous data (struct, see readGenericEEG)
%      mrk    - struct for class markers (alternativ: just pos vector)
%      ival   - time interval relative to marker [start ms, end ms]
%      jitter - vector of time shifts (jitter)
%
% OUT  epo       structure of epoched signals
%         .x     - signals (time x channels x epochs)
%         .y     - class labels (if y is not a field of mrk)
%         .t     - time axis
%         .fs    - sampling interval
%         .clab  - channel labels
%         .title - title of data set (if title is a field of cnt)
%         .[...] - other fields are copied from cnt
%
% SEE  readGenericEEG, makeClassMarkers

warning('function is obsolete: use makeEpochs instead');

if ~isstruct(mrk), 
  mrk.pos= mrk;
elseif cnt.fs~=mrk.fs, 
  error('cnt data and class markers have different sampling interval');
end

if exist('jitter', 'var'),
  epo= [];
  for ij= 1:length(jitter),
    epo_append= makeSegments(cnt, mrk, ival+jitter(ij));
    epo= proc_appendEpochs(epo, epo_append);
  end
  epo.nJits= length(jitter);
  return
end


iv= getIvalIndices(ival, cnt.fs);
t= length(iv);
nChans= size(cnt.x, 2);
nEvents= length(mrk.pos);
valid= find(iv(1)+mrk.pos>0 & iv(end)+mrk.pos<size(cnt.x,1));
if length(valid)<nEvents,
  warning(sprintf('%d segments dropped', nEvents-length(valid)));
  nEvents= length(valid);
end
IV= iv(:)*ones(1,nEvents) + ones(t,1)*mrk.pos(valid);

epo= copyStruct(cnt, 'x');
epo.x= reshape(cnt.x(IV, :), [t nEvents, nChans]);
epo.x= permute(epo.x, [1 3 2]);
if isfield(mrk, 'y'), 
  epo.y= mrk.y(:,valid);
end
epo.t= linspace(ival(1), ival(2), length(iv));

if isfield(mrk, 'className'),
  epo.className= mrk.className;
end
