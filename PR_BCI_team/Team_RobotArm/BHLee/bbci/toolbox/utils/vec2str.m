function str= vec2str(v, fmt, inbetween)
%str= vec2str(v, <format, inbetween=', '>)
%
% v can be a vector of real numbers or a cell array of strings.
% the default for format is '%d' in the former and '%s' in the latter case.

if ~exist('fmt', 'var') | isempty(fmt), 
  if iscell(v) || ischar(v),
    fmt='%s'; 
  else
    fmt='%d'; 
  end
end
if ~exist('inbetween', 'var'), inbetween=', '; end

if isempty(v),
  str= '[]';
elseif iscell(v),
  isprint= 1:length(v);
  for vi= 1:length(v),
    if ischar(v{vi}),
      fstr{vi}= ['%s' inbetween];
    elseif ~iscell(v{vi}) & ~isstruct(v{vi}),
      fstr{vi}= repmat([fmt inbetween], 1, length(v{vi}));
    else
      isprint= setdiff(isprint, vi);
    end
  end
  if isempty(isprint), str=''; return; end
  v= {v{isprint}};
  fstr= {fstr{isprint}};
  if length(isprint)>1,
    str= [sprintf([fstr{1:end-1}], v{1:end-1}) ...
          sprintf(fstr{end}(1:end-length(inbetween)), v{end})];
  else
    str= sprintf(fstr{end}(1:end-length(inbetween)), v{end});
  end
else
  if length(v)==1 || ischar(v),
    str= sprintf(fmt, v);
  else
    str= [sprintf([fmt inbetween], v(1:end-1)) ...
          sprintf(fmt, v(end))];
  end
end
