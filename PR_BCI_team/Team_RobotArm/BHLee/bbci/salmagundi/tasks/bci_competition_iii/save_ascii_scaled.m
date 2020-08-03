function fid= save_ascii_scaled(file_name, x, fmt, scale)
% save_ascii - Saves a 2D array as ASCII file.
%
% Synopsis:
%   RET = save_ascii(FILE_NAME, X)
%   RET = save_ascii(FILE_NAME, X, FMT)
%   RET = save_ascii(FILE_NAME, X, SCALE)
%
% Arguments:
%   FILE_NAME: name of the file to save. If it does not contain a '.',
%        the appendix '.txt' is appended.
%   X:   variable
%   FMT: format for printing the elements of the array. This argument
%        is passed to the function FPRINTF. The default value is '%d'
%        for integer data types, '%.8e' for single and '%.16e' for
%        double valued arrays.
%   SCALE: data is multiplied by this value before saveing, default is 1.
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
%
% See also: save, fprintf

% benjamin.blankertz@first.fhg.de, Nov 2004

if isempty(x),
  error('I refuse to write empty files.');
end

if ~ismember('.', file_name),
  file_name= strcat(file_name, '.txt');
end
if nargin<3,
  if isa(x, 'uint8') | isa(x, 'uint16') | isa(x, 'uint32') | ...
        isa(x, 'int8') | isa(x, 'int16') | isa(x, 'int32'),
    fmt= '%d';
  elseif isa(x, 'single'),
    fmt= '%.8e';
  else
    fmt= '%e';
  end
end
if nargin<4,
  scale= 1;
end

fid= fopen(file_name, 'w');
if fid==-1,
  return
end
for ii= 1:size(x, 1),
  fprintf(fid, fmt, double(x(ii,1))*scale);
  fprintf(fid, ['\t' fmt], double(x(ii,2:end))*scale);
  fprintf(fid, '\n');
end
fclose(fid);
