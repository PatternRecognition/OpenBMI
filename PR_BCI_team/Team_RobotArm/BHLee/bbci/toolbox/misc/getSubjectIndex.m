function [idx, code]= getSubjectIndex(subbase, subject, codeonly)
%[idx, code]= getSubjectIndex(subbase, subject)
%
%code= getSubjectIndex(subbase, subject, 'code');
%
% C  readDatabase

idx= strmatch(subject, {subbase.name}, 'exact');
if isempty(idx),
  error(sprintf('subject %s unknown', subject));
end
code= subbase(idx).code;

if exist('codeonly', 'var') & isequal(codeonly, 'code'),
  idx= code;
  clear code;
end
