function out= pyff_loadSettings(file)
%PYFF_LOADSETTINGS - Load variable settings from a JSON file
%
%OUT= pyff_loadSettings(FILE)
%
%Arguments:
% FILE: Filename of the json file. Suffix '.json' is appended is no
%    suffix is given.
%
%Output:
% OUT: Struct containing all variables of the JSON file as fields.

global BCI_DIR

if ~exist('p_json', 'file'),
  addpath([BCI_DIR 'import/json']);
end

if ~ismember('.', file),
  file= strcat(file, '.json');
end
if ~isabsolutepath(file),
  file= strcat([BCI_DIR 'acquisition/setups/'], file);
end

fid= fopen(file, 'rt'); 
if fid==-1,
  error(sprintf('file <%s> could not be opened.', file));
end
inString = fscanf(fid,'%c'); 
fclose(fid);
out= p_json(inString);
