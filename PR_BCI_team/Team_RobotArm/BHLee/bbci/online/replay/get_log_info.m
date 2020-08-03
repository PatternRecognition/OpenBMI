function [mr,pos] = get_log_info(nam,fil);

global LOG_DIR;

fid = fopen([LOG_DIR 'feedback_' nam '_' int2str(fil) '.log'],'r');

mr = [];
pos = [];

while ~feof(fid);
  s = fgets(fid);
  if ~isempty(strmatch('MARKER',s));
    s = s(10:end);
    c = strfind(s,'§');
    mr = [mr,str2num(s(1:c(1)-1))];
    s = s(c(1)+1:end);
    c = strfind(s,'=');
    s = s(c(1)+1:end);
    pos = [pos,str2num(s)];
  end
end

ind = find(mr>=200);

mr = mr(ind);
pos = pos(ind);




  