function p = toolboxpath(d)
%GENPATH Generate recursive toolbox path.
%   P = GENPATH returns a new path string by adding all the subdirectories 
%   of MATLABROOT/toolbox, including empty subdirectories. 
%
%   P = GENPATH(D) returns a path string starting in D, plus, recursively, 
%   all the subdirectories of D, including empty subdirectories.
%   
%   NOTE 1: GENPATH will not exactly recreate the original MATLAB path.
%
%   NOTE 2: GENPATH only includes subdirectories allowed on the MATLAB
%   path.
%
%   See also PATH, ADDPATH, RMPATH, SAVEPATH.

%   Copyright 1984-2006 The MathWorks, Inc.
%   $Revision: 1.13.4.5 $ $Date: 2008/06/24 17:12:06 $
%------------------------------------------------------------------------------
%
%   modified by marton to exclude all .svn directories
%

global BCI_DIR
if nargin==0
    d = [BCI_DIR 'toolbox'];
end

% initialise variables
classsep = '@';  % qualifier for overloaded class directories
packagesep = '+';  % qualifier for overloaded package directories
p = '';           % path to be returned

% Generate path based on given root directory
files = dir(d);
if isempty(files)
  return
end

% Add d to the path even if it is empty.
p = [p d pathsep];

% set logical vector for subdirectory entries in d
isdir = logical(cat(1,files.isdir));
%
% Recursively descend through directories which are neither
% private nor "class" directories.
%
dirs = files(isdir); % select only directory entries from the current listing

for i=1:length(dirs)
   dirname = dirs(i).name;
   if    ~strcmp( dirname,'.')           && ...
         ~strcmp( dirname,'..')          && ...
         ~strcmp( dirname,'.svn')        && ...
         ~strncmp( dirname,classsep,1)   && ...
         ~strncmp( dirname,packagesep,1) && ...
         ~strcmp( dirname,'private')
      p = [p toolboxpath(fullfile(d,dirname))]; % recursive calling of this function.
   end
end

%------------------------------------------------------------------------------
