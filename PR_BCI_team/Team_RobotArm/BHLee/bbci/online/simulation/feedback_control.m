function feedback_control(inifile)
%FEEDBACK_CONTROL allows controlling a feedbacks via UDP.
%
% usage:
%   feedback_control(propertylist)
%   feedback_control(inifile)
%
% input:
%   propertylist:  - a struct with fields
%                      machine -  name of a machine
%                      port - a port
%                      fs - a sampling rate
%                      protocol - the protocol as numeric or cell
%                      array
%                      position - the position of the gui
%                      fields - cell array describing the parts of
%                      the communication
%   inifile:       - a name of a inifile. This should be a script
%                    which define propertylist 
%
% GUIDO DORNHEGE, 18/02/2004

%persistent propertylist fig fig_obj bn bln cp
global feedback_control_status
global BCI_DIR
global general_port_fields


if ischar(inifile)
  if filesep~=inifile(1)
    inifile = [BCI_DIR 'bbci_bet/simulation/setups/' inifile];
  end
  if exist(inifile,'file') | exist([inifile '.m'],'file')
    c = strfind(inifile,'/');
    c = c(end);
    d = cd;
    cd(inifile(1:c(1)));
    eval(inifile(c(1)+1:end));
    cd(d);
  else
    error('inifile does not exist');
  end
elseif isstruct(inifile)
  propertylist = inifile;    
else
  error('input not understand');
end

propertylist = set_defaults(propertylist,...
			   'machine',general_port_fields(1).graphic{1},...
			   'port',12489,...
			   'fs',25,...
			   'position',[100 100 400 300],...
                           'fields',{});


try
  send_data_udp;
end
try
  close(fig);
end
fig = [];



fig = figure;
height = propertylist.position(4)/((length(propertylist.fields))+2);
tabs = 0.4*propertylist.position(3);

set(fig,'MenuBar','none');
set(fig,'CloseRequestFcn',['closereq;global feedback_control_status;' ...
		    'feedback_control_status = 1;']);
  
set(fig,'position',propertylist.position);
% $$$ set(fig,'WindowButtonMotionFcn',['global feedback_control_status;' ...
% $$$ 		    ' feedback_control_status=[5];']);
% $$$ set(fig,'WindowButtonUpFcn',['global feedback_control_status;' ...
% $$$ 		    ' feedback_control_status=[];']);


bbn = {};
bli = [];
sli = [];
fig_obj = zeros(2,length(propertylist.fields));
machine = uicontrol(fig,'Style','Text');
set(machine,'String','Machine','Position',[0,propertylist.position(4)-height,tabs,height]);
machine2 = uicontrol(fig,'Style','edit');
set(machine2,'String',propertylist.machine);

set(machine2,'position',[tabs,propertylist.position(4)-height,propertylist.position(3)-tabs,height]);
set(machine2,'Callback','global feedback_control_status; feedback_control_status = 17;');

port = uicontrol(fig,'Style','Text');
set(port,'String','Port','Position',[0,propertylist.position(4)-2*height,tabs,height]);
port2 = uicontrol(fig,'Style','edit');
set(port2,'String',propertylist.port);

set(port2,'position',[tabs,propertylist.position(4)-2*height,propertylist.position(3)-tabs,height]);
set(port2,'Callback','global feedback_control_status; feedback_control_status = 17;');

for i = 1:length(propertylist.fields)
  fig_obj(1,i) = uicontrol(fig,'Style','Text');
  set(fig_obj(1,i),'String',propertylist.fields{i}{1});
  set(fig_obj(1,i),'position',[0, propertylist.position(4)-(i+2)*height,tabs,height]);
  switch propertylist.fields{i}{2}
   case 'te'
    fig_obj(2,i) = uicontrol(fig,'Style','Text','String',propertylist.fields{i}{3});
    bbn = {bbn{:},'String'};
   case 'sl'
    if length(propertylist.fields{i})==2
      propertylist.fields{i}{3} = [-1 1];
    end
    if length(propertylist.fields{i})==3
      propertylist.fields{i}{4} = [0.01 0.1];
    end
    ob = size(bli,2)*2+1;
    set(fig_obj(1,i),'String',[propertylist.fields{i}{1}, ' ', num2str(ob),'-',num2str(ob+1)]);
    bli = [bli,[ob;ob+1;propertylist.fields{i}{4}(1)]];
    sli = [sli,i];
    fig_obj(2,i) = uicontrol(fig,'Style','Slider','Min', ...
			     propertylist.fields{i}{3}(1),'Max', ...
			     propertylist.fields{i}{3}(2),'SliderStep', ...
			     propertylist.fields{i}{4},'Value', ...
			     mean(propertylist.fields{i}{3}), ...
			     'Tooltipstring',num2str(mean(propertylist.fields{i}{3})));
    bbn = {bbn{:},'Value'};
    set(fig_obj(2,i),'Callback','set(gcbo,''Tooltipstring'',num2str(get(gcbo,''Value'')))');
   case 'fl'
    fig_obj(2,i) = uicontrol(fig,'Style','togglebutton','Value',0, ...
			     'String','false','Callback',...
			     ['set(gcbo,''String'',getfield({''false'',''true''},{1+get(gcbo,''Value'')}));']);
    bbn = {bbn{:},'Value'};
    
   case 'ed'
    if length(propertylist.fields{i})>2
      fig_obj(2,i) = uicontrol(fig,'Style','edit','String', ...
			       propertylist.fields{i}{3});
    else
      fig_obj(2,i) = uicontrol(fig,'Style','edit','String', '0');
    end
    bbn = {bbn{:},'String'};
  end
  set(fig_obj(2,i),'position',[tabs,propertylist.position(4)-(i+2)*height,propertylist.position(3)-tabs,height]);
  
end

set(fig,'ResizeFcn',['global feedback_control_status;' ...
		    'feedback_control_status = 2;']);
set(fig,'BusyAction','cancel');

feedback_control_status = 0;



send_data_udp(propertylist.machine,propertylist.port);
waitForSync;

package = zeros(1,length(bbn)+4);

while feedback_control_status(1)~=1
  package(2) = package(2)+1;
  package(3) = package(3)+1000/propertylist.fs;
  si = get(fig,'CurrentCharacter');
  si = str2num(si);
  if ~isempty(si)
    [i,j] = find(bli(1:2,:)==si);
    if ~isempty(j)
      di = get(fig_obj(2,sli(j(1))),'Value')+bli(3,j(1))*(i(1)*2-3);
      di = max(min(di,get(fig_obj(2,sli(j(1))),'Max')),get(fig_obj(2,sli(j(1))),'Min'));
      set(fig_obj(2,sli(j(1))),'Value',di);
      set(fig,'CurrentCharacter',' ');
    end
  end
  for i = 1:length(bbn)
    s = get(fig_obj(2,i),bbn{i});
    if ischar(s)
      s = str2num(s);
    end
    package(i+4) = s;
  end
%   if feedback_control_status(1) ==2
%     pos = get(fig,'position');
%     height = pos(4)/(length(propertylist.fields));
%     tabs = 0.4*pos(3);
%     for i = 1:length(propertylist.protocol);
%       set(fig_obj(1,i),'position',[0, pos(4)-i*height,tabs,height]);
%       set(fig_obj(2,i),'position',[tabs,pos(4)-i*height,pos(3)-tabs,height]);
%     end  
%   end

  if feedback_control_status(1)== 17
    send_data_udp;err 10040
    propertylist.machine = get(machine2,'String');
    propertylist.port = str2num(get(port2,'String'));
    send_data_udp(propertylist.machine,propertylist.port);
    fprintf('Connect to %d at %s\n',propertylist.port,propertylist.machine);
  end
  propertylist.fs
  feedback_control_status = 0;
  drawnow
  package
  send_data_udp(package);
  clock
  propertylist.fs
  waitForSync(1000/(propertylist.fs*2));
end

send_data_udp;


