function str = get_eval_string(var,begin);
% GET_EVAL_STRING ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% return an evaluable string such that var is turned into the variable begin
% 
% usage:
%    str = get_eval_string(var,begin);
% 
% input:
%    var     the values to be turned into the variable
%    begin   the name of the variable informations should be turned into
%    
% output:
%    str     the evaluable string
%
% Guido Dornhege
% $Id: get_eval_string.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

if ~exist('begin','var')
  begin = '';
end

% do it recursively for structs, ignoring leading '.'
if isstruct(var)
  str = '';
  fi = fieldnames(var);
  for i = 1:length(fi)
    for j = 1:length(var)
      if length(begin)>0
        str = [str,get_eval_string(getfield(var(j),fi{i}),[begin '(' int2str(j) ').' fi{i}])];
      else
        str = [str,get_eval_string(getfield(var,fi{i}),[begin fi{i}])];
      end
    end
  end
else
  % thats simple
  str = sprintf('%s = %s;',begin,get_text_string(var));
end
