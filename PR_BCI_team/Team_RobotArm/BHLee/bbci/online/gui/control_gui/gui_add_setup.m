function [str classes] = gui_add_setup(fig,dire,player,machine,port);
% GUI_ADD_SETUP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% gui for add a setup 
%
% usage:
%    str = gui_add_setup(dire,machine,port);
% 
% input:
%    fig     the gui handle
%    dire    start directory
%    machine name of the machine
%    port    portnumer
%
% output:
%    str    empty (no new one) or full path
%
% Guido Dornhege
% $Id: gui_add_setup.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

global add_a_setup nice_gui_font

p = get(fig,'position');

f = figure;

set(f,'NumberTitle','off','Menubar','none','Name','Add a setup','Position',p,'CloseRequestFcn',sprintf('global add_a_setup; add_a_setup = -%g;',fig));

exi = uicontrol('Style','pushbutton','units','normalized','position',[0.1 0 0.3 0.05],'String','Cancel');
set(exi,'Tooltipstring','Cancel this gui');
set(exi,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.3);
set(exi,'Callback',sprintf('global add_a_setup; add_a_setup = [%g,0];',fig));
set(exi,'UserData',fig);


oki = uicontrol('Style','pushbutton','units','normalized','position',[0.6 0 0.3 0.05],'String','OK');
set(oki,'Tooltipstring','Use the file if possible');
set(oki,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.3);
set(oki,'Callback',sprintf('global add_a_setup; add_a_setup = [%g,1];',fig));
set(oki,'UserData',fig);

direc = uicontrol('Style','edit','units','normalized','position',[0 0.92 1 0.08],'String',dire);
set(direc,'Tooltipstring','Use this directory');
set(direc,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.4);
set(direc,'Callback',sprintf('global add_a_setup; add_a_setup = [%g,2];',fig));
set(direc,'UserData',fig);


[directories,files,players,classes] = get_directory_info(f,dire,player,machine,port);

dd = uicontrol('Style','listbox','units','normalized','position',[0 0.07 0.3 0.83]);
set(dd,'Tooltipstring','Available directories');
set(dd,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.02);
set(dd,'Callback',sprintf('global add_a_setup; add_a_setup = [%g,3];',fig));
set(dd,'UserData',fig);
set(dd,'Max',1,'Min',0);

ff = uicontrol('Style','listbox','units','normalized','position',[0.33 0.07 0.67 0.83]);
set(ff,'Tooltipstring','Available setups');
set(ff,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.02);
set(ff,'UserData',fig);
set(ff,'Max',1,'Min',0);

if isnumeric(directories)
  directories = {};
  files = {};
  players = [];
  classes = {};
end

visualize_graphics(dd,ff,directories,files,players,classes);
        

add_a_setup = [0,0];
while true;
  pause(0.01);
  if add_a_setup(1)==fig
    if add_a_setup(2)==0
      str = '';
      delete(f);
      return;
    end
    if add_a_setup(2)==1
      i = get(ff,'Value');
      if (i>0) && (i<=length(files))
        str = [dire,files{i}];
        classes = classes{i};
      else
        str = '';
      end
      delete(f);
      return;
    end
    if add_a_setup(2)==2
      d = get(direc,'String');
      if length(d)==0 || d(end)~=filesep
        d = [d,filesep];
        set(direc,'String',d);
      end
      [a,b,c,d] = get_directory_info(f,d,player,machine,port);
      if ~isnumeric(a)
        directories = a;
        files = b;
        players = c;
        classes = d;
        dire = get(direc,'String');
        visualize_graphics(dd,ff,directories,files,players,classes);
      else
        set(direc,'String',dire);
      end
    end
    if add_a_setup(2)==3
      d = get(direc,'String');
      e = get(dd,'Value');
      e = directories{e};
      if strcmp(e,'..')
        c = find(ismember(d,'/\'));
        if c(end)==length(d),
          c(end)= [];
        end
        d = d(1:c(end));
      elseif e~='.'
        d = [d,e,'/'];
      end
      
      [a,b,c1,c2] = get_directory_info(f,d,player,machine,port);
      if ~isnumeric(a)
        directories = a;
        files = b;
        players = c1;
        classes = c2;
        dire = d;
        visualize_graphics(dd,ff,directories,files,players,classes);
      end
      set(direc,'String',dire);
    end
  end
  add_a_setup = [0,0];
end



return;


function visualize_graphics(dd,ff,directories,files,players,classes);

set(dd,'String',directories,'Value',1);

for i = 1:length(files)
  files{i} = sprintf('%s (Player: %d, Classes: %s)',files{i},players(i),classes{i});
end

set(ff,'String',files,'Value',1);

return





function [directories,files,players,classes] = get_directory_info(f,dire,player,machine,port);

set(f,'Visible','off');
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
while isempty(a) && toc<2
  a = get_data_udp(gui_add_back,0);
  pause(0.05);
end

a = char(a);

% close the port
get_data_udp(gui_add_back);

if isempty(a) || strcmp(a,'-1');
  set(f,'Visible','on');
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
while po<=length(str) && ~isempty(str{po})
  directories = {directories{:},str{po}};
  po = po+1;
end

po = po+1;

while po<=length(str) && ~isempty(str{po})
  c = strfind(str{po},'-');
  files = {files{:},str{po}(1:c(1)-2)};
  rs = str{po}(c(1)+2:end);
  c = strfind(rs,'-');
  players = [players,str2num(rs(1:c(1)-2))];
  rs = rs(c(1)+2:end);
  classes = {classes{:},rs};
  po = po+1;
end


set(f,'Visible','on');


