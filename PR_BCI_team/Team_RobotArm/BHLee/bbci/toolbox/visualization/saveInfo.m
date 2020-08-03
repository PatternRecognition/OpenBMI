function saveInfo(file, vec, varargin)
%saveInfo(fileName, vec, <params>)
%
% if fileName does not include an absolute path, the global variable
% TEX_DIR is prepended. The suffix '_info.txt' is appened, an the given
% vector (or cell array) is printed to that file using vec2str called
% with optional params.
%
% SEE vec2str
% GLOBZ TEX_DIR


if file(1)==filesep | (~isunix & file(2)==':'),
  fullName = file;
else
  global TEX_DIR
  fullName= [TEX_DIR file];
end

fid= fopen([fullName '_info.txt'], 'wt');
if fid==-1,
  error(sprintf('could not open <%s> for writing', [fullName '_info.txt']));
end

fprintf(fid, vec2str(vec, varargin{:}));
fclose(fid);
