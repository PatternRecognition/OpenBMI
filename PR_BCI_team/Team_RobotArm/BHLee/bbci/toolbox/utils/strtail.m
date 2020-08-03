function cstr= strtail(cstr,varargin)
%tail= strtail(str,<delim>)
%
% returns the string after (not including) the first space
% works also for cells of strings
% 'delim' -  delimiter of the string head, e.g. '_' (default ' ')
%
% See also strhead

if nargin<2
    delim = ' ';
else
    delim = varargin{1};
end

if ~iscell(cstr), 
  cstr= strtail({cstr},delim); 
  cstr= cstr{1};
  return
end

for il= 1:length(cstr),
  cstr{il}= cstr{il}(max([1 findstr(delim, cstr{il})+1]):end);
end
