function [directories,files,players,classes] = get_directory_info(f,dire,player,machine,port);

if ~isempty(f),set(f,'Visible','off');end
drawnow;
files = {};
players = [];
classes = {};

% open a listening port
portback = 12400+player;
ma = get_hostname;

gui_add_back = get_data_udp(portback);

send_evalstring(machine,port,sprintf('get_directory_informations = {''%s'',''%s'',%d};',dire,ma,portback));

% GET THE INFO
tic;
a = [];
while isempty(a) & toc<2
  a = get_data_udp(gui_add_back,0);
  pause(0.05);
end

a = char(a);

% close the port
get_data_udp(gui_add_back);

if isempty(a) | strcmp(a,'-1');
  if ~isempty(f),set(f,'Visible','on');end
  directories = -1;
  return;
end

ind = [0,find(double(a)==10)];
str = cell(1,length(ind)-1);
for i = 1:length(ind)-1
  str{i} = a(ind(i)+1:ind(i+1)-1);
end

directories = {};

po = 1;
while po<=length(str) & ~isempty(str{po})
  directories = {directories{:},str{po}};
  po = po+1;
end

po = po+1;

while po<=length(str) & ~isempty(str{po})
  c = strfind(str{po},'-');
  files = {files{:},str{po}(1:c(1)-2)};
  rs = str{po}(c(1)+2:end);
  c = strfind(rs,'-');
  players = [players,str2num(rs(1:c(1)-2))];
  rs = rs(c(1)+2:end);
  classes = {classes{:},rs};
  po = po+1;
end


if ~isempty(f),set(f,'Visible','on');end
