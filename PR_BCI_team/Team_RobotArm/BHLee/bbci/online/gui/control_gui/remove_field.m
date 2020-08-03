function s = remove_field(s,str);
%REMOVE_FIELD ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% deletes the information str in s (compared to rmfield it is possible to delete in a.b.c = struct('d',1,'e',2); the field a.b.c.d by remove_field(a,'b.c.d')
%
% usage:
%    s = remove_field(s,str);
%
% input:
%    s   the variable
%    str the string to delete
%
% Guido Dornhege
% $Id: remove_field.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% solve it recursively
c = strfind(str,'.');

if isempty(c)
  % no further struct
  s = rmfield(s,str);
else
  a = remove_field(getfield(s,str(1:c(1)-1)),str(c(1)+1:end));
  if ~isempty(fieldnames(a))
    s = setfield(s,str(1:c(1)-1),a);
  else
    s = rmfield(s,str(1:c(1)-1));
  end
end
