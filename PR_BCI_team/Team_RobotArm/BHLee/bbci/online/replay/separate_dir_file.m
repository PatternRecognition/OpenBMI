function [file,dire] = separate_dir_file(file);

if isunix
  c = strfind(file,'/');
  if isempty(c)
    dire = '';
  else
    dire = file(1:c(end));
    file = file(c(end)+1:end);
  end
else
  c = strfind(file,'\');
  if isempty(c)
    dire = '';
  else
    dire = file(1:c(end));
    file = file(c(end)+1:end);
  end
end
