function opt = setdefault(vals,fields,defaults);
%SETDEFAULT takes the entries of cell array vals and builds a struct with fields fields with values vals{i} or if not exist or isempty takes the default values.
%
% usage:
%     opt = setdefault({val1,...},{field1,...},{defaults1,...});
% 
% input:
%     val1    a usual value
%     field1  a field name
%     defaults1   another value
%
% output:
%     opt    a struct with opt.fieldn = valn, if exist(valn) & ~isempty(valn), otherwise opt.fieldn = defaultsn;
%
% Guido Dornhege, 27/04/2004

opt = struct;
for i = 1:length(fields)
  if length(vals)<i | isempty(vals{i})
    opt = setfield(opt,fields{i},defaults{i});
  else
    opt = setfield(opt,fields{i},vals{i});
  end
end