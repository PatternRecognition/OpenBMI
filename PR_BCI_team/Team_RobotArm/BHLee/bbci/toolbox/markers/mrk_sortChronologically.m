function mrk= mrk_sortChronologically(mrk, varargin)
%MRK_SORTCHRONOLOGICALLY - Sort Markers Chronologically
%
%Synopsis:
% MRK_OUT= mrk_sortChronologically(MRK_IN, <OPT>)
%
%Obsolete (but still working) Synopsis:
% MRK_OUT= mrk_sortChronologically(MRK_IN, <classWise>, <OPT>)
%
%Arguments:
% MRK_IN: Marker structure as received by eegfile_loadBV
% OPT: struct or property/value list of optional properties:
%  'removevoidclasses': Void classes are removed from the list of classes.
%  'classwise': Each class is sorted chronologically, default 0.
%
%Returns:
% MRK_OUT: Marker structure with events sorted chronologically

% blanker@cs.tu-berlin.de

if length(varargin)>0 & isnumeric(varargin{1}),
  opt= struct('classwise', varargin{1});
  varargin= varargin{2:end};
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'classwise', 0);

if opt.classwise,
  nClasses= size(mrk.y,1);
  si= zeros(nClasses, length(mrk.pos));
  for ci= 1:nClasses,
    idx= find(mrk.y(ci,:));
    [so,sidx]= sort(mrk.pos(idx));
    si(ci, 1:length(idx))= idx(sidx);
  end
  si= si(find(si));
else
  [so,si]= sort(mrk.pos);
end

mrk= mrk_chooseEvents(mrk, si, opt, 'sort',0);
