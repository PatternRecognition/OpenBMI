function labels = expbase2labels(eb, varargin)
% labels = expbase2labels(eb, varargin)
% Ryota Tomioka, 2007
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'classes', '');

labels = cell(size(eb));

for i=1:prod(size(eb))
  if isfield(eb,'classes')
    if ischar(eb(i).classes)
      classes = eb(i).classes;
    else
      classes = getShortClassname(eb(i).classes);
    end
  else
    classes = opt.classes;
  end
  
  labels{i} = [eb(i).subject, '_',...
               eb(i).date, '_',...
               eb(i).paradigm, '_', classes];
end

if length(labels)==1
  labels = cell2mat(labels);
end
