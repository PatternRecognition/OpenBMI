function [epo, valid]= makeEmbededEpochs(cnt, mrk, ival, lags)
%epo= makeEpochs(cnt, mrk, ival, <lags>)
%
% IN   cnt    - continuous data (struct, see readGenericEEG)
%      mrk    - struct for class markers (alternativ: just pos vector)
%      ival   - time interval relative to marker [start ms, end ms]
%      lags   - vector of eg. time shifts (lagged coordinates) [ms]
%               e.g. [2 4 6 10] ==> x_t, x_t-2, ... , x_t-10
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
% the structure 'mrk' may contain a field 'indexedByEpochs' being a
% cell array of field names of mrk. in this case those fields are 
% copied to the epo structure.
%
% SEE  readGenericEEG, makeClassMarkers

% original version "makeEpochs" by bb 02/03, ida.first.fhg.de 
% extended by stl 06/04, ida.first.fhg.de
  
if ~exist('copy_fields','var'),
  copy_fields= {'className', 'equi', 'indexedByEpochs'};
end


if ~isstruct(mrk), 
  mrk.pos= mrk;
elseif cnt.fs~=mrk.fs, 
  error('cnt data and class markers have different sampling intervals');
end

nEvents= length(mrk.pos);
iv= getIvalIndices(ival, cnt.fs);
is= iv(1);
ie= iv(end);

valid= find(is+mrk.pos>0 & ie+mrk.pos<=size(cnt.x,1));
if length(valid)<nEvents,
  mrk= mrk_selectEvents(mrk, valid);
  warning(sprintf('%d segments dropped', nEvents-length(valid)));
end
nEvents= length(valid);


if exist('lags', 'var'),
  epo = makeEpochs(cnt, mrk, ival) ;
  [T, nChans, nEvt]  = size(epo.x) ;
  for tau= lags,
    epo_append= makeEpochs(cnt, mrk, ival-tau);
    for ch = 1: nChans,
      epo_append.clab{ch} = [epo_append.clab{ch} ' lag=' num2str(tau)] ;
    end ;
    epo= proc_appendChannels(epo, epo_append);
  end
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
