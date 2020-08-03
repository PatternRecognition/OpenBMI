function bool= isabsolutepath(file)
% ISABSOLUTEPATH - Determines whether a path is absolute or relative.
%
% Synopsis:
%   bool = isabsolutepath(FILE)
%
% Returns: 1 (absolute path) or 0 (relative path)
%
bool= (isunix &&  file(1)==filesep) || (ispc && file(2)==':');
