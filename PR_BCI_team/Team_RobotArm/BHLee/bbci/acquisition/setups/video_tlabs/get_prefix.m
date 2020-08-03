
function s = get_prefix(s,prefix)
  if nargin<2
    prefix = '_';
  end
  idx = find(s==prefix);
  s = s(1:idx(1)-1);