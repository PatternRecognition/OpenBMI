function str= strrep_globals(str, replace_list)

if nargin<2,
  replace_list= {'TODAY_DIR', 'VP_CODE', 'TMP_DIR'};
end

eval(['global ' sprintf('%s ', replace_list{:})]);
for k= 1:length(replace_list),
  name= replace_list{k};
  if isempty(eval(name)),
    if ~isempty(strfind(['$' name], str)),
      error(sprintf('string contains ''$%s'', but this global variable is undefined', name));
    end
  else
    str= strrep(str, ['$' name], eval(name));
  end
end
