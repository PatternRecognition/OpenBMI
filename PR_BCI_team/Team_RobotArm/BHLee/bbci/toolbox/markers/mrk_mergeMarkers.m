function mrk= mrk_mergeMarkers(mrk1, mrk2, varargin)
%MRK_MERGEMARKERS - Merge Marker Structs
%
%
%Description:
% This function merges two or more marker structs into one.
%
%Synopsis:
% MRK= mrk_mergeMarkers(MRK1, MRK2, ...)
%
% OPT
% keepFields    - keeps other fields  even if found in only one of the marker structs (default 1)

opt = [];
for ii=1:numel(varargin)
  if ischar(varargin{ii}) % Not a marker struct
    opt= propertylist2struct(varargin{ii:end});
    varargin = varargin(1:ii-1);
    break
  end
end
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'keepFields',1);

if isempty(mrk1),
  mrk= mrk2;
  return;
elseif isempty(mrk2),
  mrk= mrk1;
  return;
end

if mrk1.fs~=mrk2.fs,
  error('mismatch in fs');
end

%% pos and toe
mrk.fs= mrk1.fs;
mrk.pos= cat(1, mrk1.pos(:), mrk2.pos(:))';
if isfield(mrk1, 'toe') && ~isfield(mrk2, 'toe'),
  warning('field ''toe'' only found in MRK1');
  mrk2.toe= repmat(NaN, size(mrk1.toe));
end
if isfield(mrk2, 'toe') && ~isfield(mrk1, 'toe'),
  warning('field ''toe'' only found in MRK2');
  mrk1.toe= repmat(NaN, size(mrk2.toe));
end
if isfield(mrk1, 'toe'),
  mrk.toe= cat(1, mrk1.toe(:), mrk2.toe(:))';
end

%% Indexed by epochs
if isfield(mrk1, 'indexedByEpochs'),
  to_be_removed= {};
  for Fld= mrk1.indexedByEpochs,
    fld= Fld{1};
    if isfield(mrk2, fld),
      tmp1= getfield(mrk1, fld);
      tmp2= getfield(mrk2, fld);
      di= min(find(size(tmp1)>1));  %% first nonvoid dimension
      if isempty(di), di=1; end
      mrk= setfield(mrk, fld, cat(di, tmp1, tmp2));
    else
      warning('field %s not found in second marker structure', fld);
      to_be_removed= cat(2, to_be_removed, {fld});
    end
  end
  mrk.indexedByEpochs = setdiff(mrk1.indexedByEpochs, to_be_removed);
end

%% Y field
if isfield(mrk1, 'y'),
  s1= size(mrk1.y);
  s2= size(mrk2.y);
  if isfield(mrk1, 'className') && isfield(mrk2, 'className'),
    mrk.y= [mrk1.y, zeros(s1(1), s2(2))];
    mrk2y= [zeros(s2(1), s1(2)), mrk2.y];
    mrk.className= mrk1.className;
    for ii = 1:length(mrk2.className)
      c = find(strcmp(mrk.className,mrk2.className{ii}));
      if isempty(c)
        mrk.y= cat(1, mrk.y, zeros(1,size(mrk.y,2)));
        mrk.className=  cat(2, mrk.className, {mrk2.className{ii}});
        c= size(mrk.y,1);
      elseif length(c)>1,
        error('multiple classes have the same name');
      end
      mrk.y(c,end-size(mrk2.y,2)+1:end)= mrk2.y(ii,:);
    end
  else
    mrk.y= [[mrk1.y; zeros(s2(1), s1(2))], [zeros(s1(1), s2(2)); mrk2.y]];
  end
end

%% Keep other fields
if opt.keepFields
  rmFields = {'indexedByEpochs','pos','toe','y','fs','className','T'};
  if isfield(mrk1,'indexedByEpochs'), rmFields = {rmFields{:} mrk1.indexedByEpochs{:}}; end
  flds = setdiff(fieldnames(mrk1),rmFields);
  flds = union(flds, setdiff(fieldnames(mrk2),rmFields));
  for ff=flds
    ff=ff{:};
    if isfield(mrk1,ff) && isfield(mrk2,ff)
      if ~isequal(mrk1.(ff),mrk2.(ff))
        warning('Field %s is different in the marker structs, taking the field from the first struct\n',ff)
      end
      mrk.(ff) = mrk1.(ff);
    elseif isfield(mrk1,ff)
      mrk.(ff) = mrk1.(ff);
    else
      mrk.(ff) = mrk2.(ff);
    end
  end
end

%% Recursion
if length(varargin)>0,
  mrk= mrk_mergeMarkers(mrk, varargin{1}, varargin{2:end},'keepFields',opt.keepFields);
end
