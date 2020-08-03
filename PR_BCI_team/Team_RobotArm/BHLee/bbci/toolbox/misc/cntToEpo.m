function [epo, valid]= cntToEpo(cnt, mrk, ival, varargin)
%CNTTOEPO - Segment continuous signals into epochs
%
%EPO= cntToEpo(CNT, MRK, IVAL, <OPT>)
%
%Arguments:
% CNT    - continuous data (struct, see eegfile_loadBV, eegfile_loadMatlab)
% MRK    - struct for class markers (see mrk_defineClasses)
% IVAL   - time interval relative to marker [start ms, end ms]
% OPT - struct or property/value list of optional properties:   
%  clab   - selection of channels
%  jitter - vector of time shifts (jitter) [ms]
%  save_incomplete_trials - default 0.
%
%Returns:
% EPO -  structure of epoched signals
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
% SEE  eegfile_loadBV, eegfile_loadMatlab, mrk_defineClasses

% bb 02/03, ida.first.fhg.de

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'clab', '*', ...
                 'marker_to_sample_position', 'middle', ...
                 'mtsp', [], ...
                 'save_incomplete_trials', 0, ...
                 'jitter', []);

if ~isdefault.mtsp,
  if ~isdefault.marker_to_sample_position,
    error('either set ''mtsp'' or ''marker_to_sample_position'', but not both');
  end
  opt.marker_to_sample_position= opt.mtsp;
end

copy_fields= {'className', 'equi', 'indexedByEpochs'};

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

switch(lower(opt.marker_to_sample_position)),
 case 'compatability',
  is= floor(ival(1)*cnt.fs/1000);
  ie= ceil(ival(2)*cnt.fs/1000);
 case 'middle',
  is= round(ival(1)*cnt.fs/1000);
  ie= round(ival(2)*cnt.fs/1000);
 case 'before',
  is= floor(ival(1)*cnt.fs/1000);
  ie= -1+ceil(ival(2)*cnt.fs/1000);
 case 'after',
  is= 1+floor(ival(1)*cnt.fs/1000);
  ie= ceil(ival(2)*cnt.fs/1000);
 otherwise,
  error('unknown mode for ''marker_to_sample_position''.');
end
iv= is:ie;
if ~isempty(opt.jitter),
  is= is + min(opt.jitter);
  ie= ie + max(opt.jitter);
end
nEvents= length(mrk.pos);
valid= find(is+mrk.pos>0 & ie+mrk.pos<=size(cnt.x,1));
if length(valid)<nEvents,
  if opt.save_incomplete_trials,
    if is+min(mrk.pos)<=0,
      error('trials to the past cannot be saved so far.');
    end
    T_add= ie+max(mrk.pos) - size(cnt.x,1);
    cnt.x= cat(1, cnt.x, NaN(T_add, size(cnt.x,2)));
  else
    mrk= mrk_chooseEvents(mrk, valid);
    warning(sprintf('%d segments dropped', nEvents-length(valid)));
    nEvents= length(valid);
  end
end

if ~isempty(opt.jitter),
  epo= [];
  for ij= 1:length(opt.jitter),
    epo_append= cntToEpo(cnt, mrk, ival+opt.jitter(ij), opt);
    epo_append.bidx= 1:length(mrk.pos);
    epo_append.jit= opt.jitter(ij)*ones(1,length(mrk.pos));
    epo= proc_appendEpochs(epo, epo_append);
  end
  epo= mrk_addIndexedField(epo, {'bidx','jit'});
  return
end

t= length(iv);
if max(mrk.pos)+iv(end)==size(cnt.x,1)+1,
  %% append one interpolated sample
  cnt.x= cat(1, cnt.x, 2*cnt.x(end,:)-cnt.x(end-1,:));
end
IV= round(iv(:)*ones(1,nEvents) + ones(t,1)*mrk.pos);

epo= rmfield(cnt,'x');
if isequal(opt.clab, '*'),
  epo.x= reshape(cnt.x(IV, :), [t nEvents size(cnt.x,2)]);
else
  cidx= chanind(cnt, opt.clab);
  epo.clab= cnt.clab(cidx);
  epo.x= reshape(cnt.x(IV, cidx), [t nEvents length(cidx)]);
end
clear IV

epo.x= permute(epo.x, [1 3 2]);
if isfield(mrk, 'y'), 
  epo.y= mrk.y;
end
%epo.t= linspace(ival(1), ival(2), length(iv));
epo.t= linspace(iv(1)/cnt.fs*1000, iv(end)/cnt.fs*1000, length(iv));

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
