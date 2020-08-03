function flag = checkWildcards(clab); 
%checkWildcards goes through the cell array clab and returns true 
%if there is in one string a wildcard (*,#,,,-)
%
% usage:
%     flag  = checkWildcards(clab);
%
% input:
%     clab     a cell array of strings
%     
% output:
%     flag     a boolean, is true if there exist at least wildcard 
%              in one string in clab
%
% Guido Dornhege, 02/12/2004
% TODO: extended documentation by Schwaighase 
% $Id: checkWildcards.m,v 1.1 2006/04/27 14:24:59 neuro_cvs Exp $

flag = 0;
if isempty(clab) 
  return;
end

if ~iscell(clab)
  clab = {clab};
end

wildcards = {'*','-','#',','};
for w = 1:length(wildcards)
  for i = 1:length(clab);
    flag = flag | ~isempty(strfind(clab{i},wildcards{w}));
  end
end
