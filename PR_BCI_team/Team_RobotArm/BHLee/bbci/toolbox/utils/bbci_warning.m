function state= bbci_warning(msg, id, file)
%BBCI_WARNING - Display warning message
%
%Description:
% Essentially this function is just a work-around such that one can
% use the increased functionality of the warning function of Matlab R>=13,
% while having programs that still run under Matlab R<13.
%
%Usage:
% STATE= bbci_warning(MSG, ID, <FILE>)
%
%Input:
% MSG:  (string) the warning message, or 'off' or 'query', see the
%       help of 'warning'.
% ID:   (string)
% FILE: (string) filename of the function in which the warning is produced.
%       (A function can determine its name via function 'mfilename'.)
%
%Example 1 (within a matlab function):
% bbci_warning('outer model selection sucks', 'validation', mfilename);
%
%Example 2:
% wstat= bbci_warning('off', 'validation');
% %% ... some code consciously producing 'validation' warnings ...
% bbci_warning(wstat);
%
%See also warning, mfilename

% Author(s): Benjamin Blankertz, long time ago

if ~exist('file','var'), 
  ff= ''; 
else
  ff= [file ': '];
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
