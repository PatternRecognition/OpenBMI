function fid= save_ascii(file_name, x, fmt,headerStr)
% save_ascii - Saves a 2D array as ASCII file.
%
% Synopsis:
%   RET = save_ascii(FILE_NAME, X)
%   RET = save_ascii(FILE_NAME, X, FMT, HEADERSTR)
%
% Arguments:
%   FILE_NAME: name of the file to save. If it does not contain a '.',
%        the appendix '.txt' is appended.
%   X:   variable, can be numeric, cell or cell string
%   FMT: format for printing the elements of the array. This argument
%        is passed to the function FPRINTF. The default value is '%d'
%        for integer data types, '%.8e' for single, '%.16e' for
%        double valued arrays, and '%s' for cell strings. Alternatively,
%        the format string can contain entries for each column.
%   HEADERSTR: string to be written into the first line of the file,
%        before the actual data. Files with header can be read into
%        Matlab with textread.m with the 'headerlines' option.
%
% Returns:
%   RET: -1 if an error occured, otherwise >= 0.
%
% Description:
%   This function does a similar thing as the matlab function SAVE when
%   used with the '-ascii' option. The additional feature of this funtion
%   is that it can store integer valued arrays in a nicer and more
%   compact format. In contrast to the matlab function you pass the
%   variable itself, not a string containing the name of the variable.
%   Column entries are separated by Tabs.
%
% Examples:
%   Create a comma-separated file with integers in the first column,
%   floats in the second column: 
%     X = [1 2.3; 2 17.5];
%     save_ascii('some.csv', X, '%i,%f')
%
%   Save a cell string:
%     X = {'nix', 'da'}
%     save_ascii('some.csv', X, '%s')
%   Save a mixed cell:
%     X = {1, 'nix'; 2, 'da'}
%     save_ascii('some.csv', X, '%i:%s')
%
% See also: save, fprintf, textread

% benjamin.blankertz@first.fhg.de, Nov 2004
% Anton Schwaighofer, Oct 2006

error(nargchk(2, 4, nargin));
if nargin<4,
  headerStr = [];
end
if nargin<3,
  fmt = [];
end

if isempty(x),
  error('I refuse to write empty files.');
end

if ~ismember('.', file_name),
  file_name= strcat(file_name, '.txt');
end
if isempty(fmt),
  if isa(x, 'uint8') | isa(x, 'uint16') | isa(x, 'uint32') | ...
        isa(x, 'int8') | isa(x, 'int16') | isa(x, 'int32'),
    fmt= '%d';
  elseif isa(x, 'single'),
    fmt= '%.8e';
  elseif iscellstr(x),
    fmt = '%s';
  else
    fmt= '%e';
  end
end

% Check how many output specifiers we have in the format string. If it is
% more than one, assume that there are as many output specifiers as
% columns.
% Don't forget that '%%' specifies a percent sign, not two outputs...
% (this might be incorrect for unlikely format strings like '%i%%%%%f'
nout = length(findstr(fmt, '%'))-length(findstr(fmt, '%%'));
if nout>1,
  % format string contains more than one output:
  if nout~=size(x,2),
    warning('Number of columns does not match number of outputs in format string');
  end
  formatstr = fmt;
else
  % Replicate the format string for one column
  formatstr = fmt;
  % Avoid writing a lonely TAB character at the end of the line if output has
  % only one column (loading such files leads to interesting effects in
  % textread)
  for j = 2:size(x,2),
    formatstr = [formatstr '\t' fmt];
  end
end
% Make sure there is a CR character at the very end of the format string
eol = findstr(formatstr, '\n');
if isempty(eol) | eol(end)~=length(formatstr)-1,
  formatstr = [formatstr '\n'];
end

fid= fopen(file_name, 'w');
if fid==-1,
  return
end
if ~isempty(headerStr),
  fprintf(fid, '%s\n', headerStr);
end
for ii= 1:size(x, 1),
  if iscell(x),
    fprintf(fid, formatstr, x{ii,:});
  else
    fprintf(fid, formatstr, double(x(ii,:)));
  end
end
fclose(fid);
