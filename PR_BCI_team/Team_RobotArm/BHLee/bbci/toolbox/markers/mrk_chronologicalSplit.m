function mrk_cs= mrk_chronologicalSplit(mrk, varargin)

% MRK_CHRONOLOGICALSPLIT - chronologically splits the markers into even-sized
% groups, such as an 'early' group (containing the first 50% of events)
% and a 'late' group (containing the last 50% of events).
%
% Synopsis:
%   [MRK]= mrk_chronologicalSplit(mrk, <OPT>)
%
% Arguments:
%   MRK: marker structure
%   OPT: Struct or property/value list of optional parameters:
%   'ngroups': number of groups in which the dataset is to be split
%               (default 2)
%   'appendix': cell array with labels for the groups
%               (default {'early' 'late'}, resp {'early', 'middle', 'late'})
%
% Returns:
%   MRK: updated marker structure
%
% Author: Benjamin B
% 07-2010: Documented, cleaned up (Matthias T) -> mkr_chronSplit
% 03-2011: restructred and extended for multiple input classes (BB)
%          renamed to mrk_chronologicalSplit since it's not downward compat.


opt= propertylist2struct(varargin{:});
[opt,isdefault]= set_defaults(opt, 'ngroups', 2, ...
                                   'appendix', {'early', 'late'});
if opt.ngroups==3,
  [opt,isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'appendix', {'early', 'middle', 'late'});
elseif opt.ngroups>3,
  [opt,isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'appendix', cprintf('block %d',1:opt.ngroups)');
end

mrk= mrk_sortChronologically(mrk);
mrk_cs= mrk;
nClasses= size(mrk.y,1);
if ~isfield(mrk, 'className'),
  if nClasses>1,
    mrk.className= cprintf('class %d', 1:nClasses)';
  else
    mrk_cs.className= opt.appendix;
  end
end

mrk_cs.y= zeros(nClasses*opt.ngroups, size(mrk.y,2));
for c= 1:nClasses,
  idx= find(mrk.y(c,:));  
  nEvents= length(idx);
  inter= round(linspace(0, nEvents, opt.ngroups+1));
  for g= 1:opt.ngroups,
    clidx= (c-1)*opt.ngroups+g;
    mrk_cs.y(clidx, idx(inter(g)+1:inter(g+1)))= 1;
    if isfield(mrk, 'className'),
      mrk_cs.className{clidx}= sprintf('%s (%s)', ...
                                       mrk.className{c}, opt.appendix{g});
    end
  end
end
