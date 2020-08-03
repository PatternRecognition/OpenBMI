function [mrk, ev]= mrk_chooseEvents(mrk, ev, varargin)
%mrk= mrk_chooseEvents(mrk, ev, <opts>)
%
% idx: indices of events that are to be selected
% opts: struct or property/value list of optinal properties:
%  'sort' evokes a call to mrk_sortChronologically.
%  'removevoidclasses' deletes empty classes (default 1)
%
% the structure 'mrk' may contain a field 'indexedByEpochs' being a
% cell array of field names of mrk. in this case subarrays of those
% fields are selected. here it is assumed that the last dimension
% is indexed by events (resp. epochs).

% bb 02/03, ida.first.fhg.de


if nargin>1 & ischar(ev) & strcmpi(ev,'not'),
  ev= varargin{1};
  varargin= varargin(2:end);
  invert= 1;
else
  invert= 0;
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'sort', 0, ...
                  'removevoidclasses', 1);

if isfield(opt, 'keepemptyclasses'),
  opt.removevoidclasses= ~opt.keepemptyclasses;
end

if nargin==1 |  (ischar(ev) & strcmpi(ev,'valid')),
  ev= find(any(mrk.y,1));
end

if invert,
  ev= setdiff(1:length(mrk.pos), ev);
end

mrk.pos= mrk.pos(ev);
if isfield(mrk, 'toe'),
  mrk.toe= mrk.toe(ev);
end
if isfield(mrk, 'y'),
  mrk.y= mrk.y(:,ev);
end

if isfield(mrk, 'indexedByEpochs'),
  for Fld= mrk.indexedByEpochs,
    fld= Fld{1};
    tmp= getfield(mrk, fld);
    sz= size(tmp);
    subidx= repmat({':'}, 1, length(sz));
    subidx{end}= ev;
    mrk= setfield(mrk, fld, tmp(subidx{:}));
  end
end

if opt.removevoidclasses & isfield(mrk,'y'),
  mrk= mrk_removeVoidClasses(mrk);
end

if opt.sort,
  mrk= mrk_sortChronologically(mrk, opt);
end
