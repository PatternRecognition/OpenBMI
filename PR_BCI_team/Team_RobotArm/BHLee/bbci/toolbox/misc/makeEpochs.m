function [epo, valid]= makeEpochs(cnt, mrk, ival, jitter, copy_fields)
%epo= makeEpochs(cnt, mrk, ival, <jitter>)
%
% IN   cnt    - continuous data (struct, see readGenericEEG)
%      mrk    - struct for class markers (alternativ: just pos vector)
%      ival   - time interval relative to marker [start ms, end ms]
%      jitter - vector of time shifts (jitter) [ms]
%
% OUT  epo       structure of epoched signals
%         .x     - signals (time x channels x epochs)
%         .y     - class labels (if y is not a field of mrk)
%         .t     - time axis
%         .fs    - sampling interval
%         .clab  - channel labels
%         .title - title of data set (if title is a field of cnt)
%         .[...] - other fields are copied from cnt
%         .indexedByEpochs - in case mrk.indexedByEpochs exists (see below)
%         .[...] - fields listed in mrk.indexedByEpochs are copied from mrk
%
% in case of jittering (these fields are important for xvalidation)
%         .bidx  - (base) index of root marker, from
%                  which epochs are extracted
%         .jit   - jitter [ms] (vector of doubles)
%
% the structure 'mrk' may contain a field 'indexedByEpochs' being a
% cell array of field names of mrk. in this case those fields are 
% copied to the epo structure.
%
% SEE  readGenericEEG, makeClassMarkers

% bb 02/03, ida.first.fhg.de

if ~exist('jitter', 'var'),
  jitter= [];
end
if ~exist('copy_fields','var'),
  copy_fields= {'className', 'equi', 'indexedByEpochs'};
end

if ~isstruct(mrk), 
  mrk.pos= mrk;
elseif cnt.fs~=mrk.fs, 
  error('cnt data and class markers have different sampling intervals');
end

if isequal(size(ival), [2 1]), ival= ival'; end
if ndims(ival)>2 | size(ival,2)~=2,
  error('input argument IVAL has invalid size');
end

%% redefine marker positions in case different intervals are
%% requested for different classes
if size(ival,1)>1,
  if size(ival,1)~=size(mrk.y,1),
    error('size of IVAL does not match #classes');
  end
  dd= diff(ival, 1, 2);
  if any(dd~=dd(1)),
    error('epochs must have the same length for all classes');
  end
  %% set mrk.pos such that all requested intervals are [0 len]
  for cc= 1:size(ival,1),
    shift= ival(cc,1)/1000*mrk.fs;
    idx= find(mrk.y(cc,:));
    mrk.pos(idx)= mrk.pos(idx) + shift;
  end
  ival= [0 diff(ival(1,:))];
end

nEvents= length(mrk.pos);
iv= getIvalIndices(ival, cnt.fs);
is= iv(1);
ie= iv(end);
if ~isempty(jitter),
  is= is + min(jitter);
  ie= ie + max(jitter);
end
valid= find(is+mrk.pos>0 & ie+mrk.pos<=size(cnt.x,1));
if length(valid)<nEvents,
  mrk= mrk_chooseEvents(mrk, valid);
  warning(sprintf('%d segments dropped', nEvents-length(valid)));
  nEvents= length(valid);
end


if ~isempty(jitter),
  epo= [];
  for ij= 1:length(jitter),
    epo_append= makeEpochs(cnt, mrk, ival+jitter(ij));
    epo_append.bidx= 1:length(mrk.pos);
    epo_append.jit= jitter(ij)*ones(1,length(mrk.pos));
    epo= proc_appendEpochs(epo, epo_append);
  end
  epo= mrk_addIndexedField(epo, {'bidx','jit'});
  return
end

t= length(iv);
nChans= size(cnt.x, 2);
if max(mrk.pos)+iv(end)==size(cnt.x,1)+1,
  %% append one interpolated sample
  cnt.x= cat(1, cnt.x, 2*cnt.x(end,:)-cnt.x(end-1,:));
end
IV= round(iv(:)*ones(1,nEvents) + ones(t,1)*mrk.pos);

epo= copyStruct(cnt, 'x');
epo.x= reshape(cnt.x(IV, :), [t nEvents, nChans]);
epo.x= permute(epo.x, [1 3 2]);
if isfield(mrk, 'y'), 
  epo.y= mrk.y;
end
epo.t= linspace(ival(1), ival(2), length(iv));

if isfield(mrk, 'indexedByEpochs'),
  copy_fields= cat(2, copy_fields, mrk.indexedByEpochs);
end
for Fld= copy_fields,
  fld= Fld{1};
  if isfield(mrk, fld),
    epo= setfield(epo, fld, getfield(mrk, fld));
  end
end

if isfield(epo, 'continuous'),
  epo= rmfield(epo, 'continuous');
end
