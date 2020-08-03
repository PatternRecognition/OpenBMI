function mrk= mrk_defineClasses(mk, classDef, varargin)
%MRK_DEFINECLASSES - Convert BV Markers to class defining marker structure
%
%Synopsis:
% MRK_OUT= mrk_defineClasses(MRK_IN, CLASS_DEF, <OPT>)
%
%
%Arguments:
% MRK_IN: Marker structure as received by eegfile_loadBV
% CLASS_DEF: Class array of size {2 x nClasses}. The first row
%  specifies the markers of each class, each cell being either
%  a cell array of strings or a vector of integers. The second
%  row specifies the class names, each cell begin a string.
% OPT: struct or property/value list of optional properties:
%  'removevoidclasses': Void classes are removed from the list of classes,
%     default 0.
%  'keepallmarkers': Keep also for markers, which do not belong to any
%     of the specified classes, default 0.
%
%Returns:
% MRK_OUT: Marker structure with classes define by labels
%     (fields .y and .className)
%
%Example:
% [cnt,mk]= eegfile_loadBV('Gabriel_01_07_24/selfpaced1sGabriel');
% classDef= {[65 70], [74 192]; 'left','right'};
% mrk= mrk_defineClasses(mk, classDef);
%
% classDef= {{'S 65','S 70'},{'S 74', 'S192'}; 'left','right'}
% mrk= mrk_defineClasses(mk, classDef);
% %% does the same

if length(varargin)==1,
  opt= struct('keepallmarkers', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
[opt, isdefault]= set_defaults(opt, ...
                               'removevoidclasses', 0, ...
                               'keepallmarkers', 0);

iS= ~apply_cellwise2(regexp(mk.type, 'Stimulus'),'isempty');
iR= ~apply_cellwise2(regexp(mk.type, 'Response'),'isempty');
valid= find(iS|iR);
sgn= iS-iR;
mrk.pos= mk.pos(valid);
mrk_toe= apply_cellwise2(mk.desc(valid), inline('str2double(x(2:end))','x'));
mrk.toe= sgn(valid) .* mrk_toe;
mrk.fs= mk.fs;
if isfield(mk,'time'),
  mrk.time= mk.time(valid);
  mrk= mrk_addIndexedField(mrk, 'time');
end

nClasses= size(classDef,2);
nEvents= length(valid);
mrk.y= zeros(nClasses, nEvents);
for cc= 1:nClasses,
  if isnumeric(classDef{1,cc}),
    mrk.y(cc,:)= ismember(mrk.toe, classDef{1,cc});
  elseif iscell(classDef{1,cc}),
    mrk.y(cc,:)= ismember(mk.desc(valid), classDef{1,cc});
  else
    mrk.y(cc,:)= ismember(mk.desc(valid), classDef(1,cc));
  end
end

if size(classDef,1)>1,
  mrk.className= classDef(2,:);
end

if ~opt.keepallmarkers,
  mrk= mrk_chooseEvents(mrk, 'valid', opt);
end
if opt.removevoidclasses,
  mrk= mrk_removeVoidClasses(mrk);
end