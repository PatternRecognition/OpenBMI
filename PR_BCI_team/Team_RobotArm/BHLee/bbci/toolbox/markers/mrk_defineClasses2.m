function mrk= mrk_defineClasses2(mk, classDef, varargin)
%MRK_DEFINECLASSES2 - Convert BV Markers to class defining marker structure
%
%Usage:
% MRK_OUT= mrk_defineClasses2(MRK, CLASSDEF, <OPT>)
%
%Arguments:
% MRK: BV marker structure (struct of arrays) such as return by
%          eegfile_readBVmarkers(..., 0)
% CLASSDEF: Cell array of size {1 nClasses} or {2 nClasses}. In the first
%   row each cell defines a class. If it is a string it is matched against
%   the MRK.desc entries. If it is a cell array of two strings, the first
%   is matched against MRK.type and the seconds is matched against
%   MRK.type. Matching is performed via strpattern match, i.e., wildcards
%   can be used at the beginning and/or at the end of the string.
%
% OPT: struct or property/value list of optional properties
%  'keepvoidclasses': Default 0.
%  'keepallmarkers': Keep markers which do not belong to one of the
%     specified classes. Default 0.
%  'deleteobsoletefields': Delete those fields of the BV marker structure
%     which might be obsolete now, like 'desc', 'type'. Default 1.
%
%Returns:
% MRK_OUT: marker structure with classes affiliations in field 'y'.
%
%Example:
% [cnt,mk]= eegfile_loadBV('VPgd_07_05_24/alphaVPgd');
% classDef= {{'Comment','Augen auf'}, {'Comment','Augen zu'};
%             'eyes open', 'eyes closed'};
% mrk= mrk_defineClasses(mk, classDef);
%
% classDef= {'Augen auf', 'Augen zu'};
% mrk= mrk_defineClasses(mk, classDef);
% %% defines the same classes (type can be omitted is no danger of mismatch)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'keepvoidclasses', 0, ...
                  'keepallmarkers', 0, ...
                  'deleteobsoletefields', 1);

if opt.deleteobsoletefields,
  mrk= copy_struct(mk, 'pos', 'fs');
else
  mrk= mk;
end

nClasses= size(classDef,2);
nEvents= length(mrk.pos);
mrk.y= zeros(nClasses, nEvents);
for cc= 1:nClasses,
  if iscell(classDef{1,cc}),
    idx_type= strpatternmatch(classDef{1,cc}{1}, mk.type);
    idx_desc= strpatternmatch(classDef{1,cc}{2}, mk.desc);
    mrk.y(cc,intersect(idx_type,idx_desc))= 1;
  elseif ~isempty(classDef{1,cc}),
    idx= strpatternmatch(classDef{1,cc}, mk.desc);
    mrk.y(cc,idx)= 1;
  end
end

if size(classDef,1)>1,
  mrk.className= classDef(2,:);
end

if ~opt.keepallmarkers,
  iclass= find(any(mrk.y,1));
  mrk= mrk_chooseEvents(mrk, iclass, 'keepemptyclasses',opt.keepvoidclasses);
end
