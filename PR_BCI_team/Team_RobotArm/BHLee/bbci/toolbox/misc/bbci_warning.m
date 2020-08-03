function state= bbci_warning(msg, id, varargin)
%state= bbci_warning(msg, id, <file>)
%
% msg may also be 'off' or 'query', see warning
% the canonical choice for 'file' is mfilename.
%
% essentially this function is just a work-around such that one can
% use the increase functionality of the warning function of Matlab R>=13,
% while having programs that still run under Matlab R<13.

% bb, ida.first.fhg.de

persistent id_list last_warning

if length(varargin)==1 & ischar(varargin{1}),
  opt= struct('file', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'file', '', ...
                  'interval', 10*60);

this_warning= clock;
id_idx= strmatch(msg, id_list, 'exact');
if isempty(id_idx),
  id_list= cat(2, id_list, {msg});
  last_warning= [last_warning; this_warning];
else 
  if etime(this_warning, last_warning(id_idx,:)) < opt.interval,
    return;
  end
  last_warning(id_idx,:)= this_warning;
end

if ~isempty(opt.file),
  ff= ''; 
else
  ff= [opt.file ': '];
end

a= sscanf(getfield(ver('MATLAB'), 'Release'), '(R%d)');
if a>=13,
  if isstruct(msg),
    warning(msg);
  elseif strcmp(msg, 'off'),
    if nargout>0,
      state= warning('query', ['bbci:' id]);
    end
    warning('off', ['bbci:' id]);
  elseif strcmp(msg, 'query'),
    state= warning('query', ['bbci:' id]);
  else
    warning(['bbci:' id], [ff msg]);
  end
else
  if ~isempty(msg) & isempty(strmatch(msg, {'off','query'})),
    state= warning([ff msg]);
  else
    state= [];
  end
end
