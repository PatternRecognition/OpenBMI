function [name,tool,value,visible] = ask_for_fields(fig,name,tool,value,visible);
% ASK_FOR_FIELDS ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% opens a gui to allow changes of some fields
%
% usage:
%    [name,tool,value,visible] = ask_for_fields(fig,name,tool,value,visible);
% 
% input:
%    fig     the handle of the gui
%    name    the name of the variable
%    tool    the help string of the variable
%    value   the value of the variable
%    visible flag if field is visible in gui or not
% 
% output:
%    name, tool, value, visible modified values if pressed Ok, otherwise empty
%
% Guido Dornhege
% $Id: ask_for_fields.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

global ask_for_fields nice_gui_font

if isempty(visible)
  visible = 1;
end

% plot the figure
p = get(fig,'position');

f = figure;
if isempty(name)
  str = 'Add the field';
else
  str = 'Modify the field';
end

set(f,'NumberTitle','off','Menubar','none','Name',str,'Position',p,'CloseRequestFcn',sprintf('global ask_for_fields; ask_for_fields = -%f;closereq;',fig));

% prepare the buttons
oki = uicontrol('Style','pushbutton','units','normalized','position',[0.6 0 0.3 0.2],'String','OK','Tooltipstring','Use these values','FontUnits','normalized','FontSize',0.4,'FontName',nice_gui_font,'Callback',sprintf('global ask_for_fields; ask_for_fields=%f;',fig));

can = uicontrol('Style','pushbutton','units','normalized','position',[0.1 0 0.3 0.2],'String','Cancel','Tooltipstring','Use these values','FontUnits','normalized','FontSize',0.4,'FontName',nice_gui_font,'Callback',sprintf('global ask_for_fields; ask_for_fields=-%f;',fig));

nam_text = uicontrol('Style','text','units','normalized','position',[0.1 0.82 0.3 0.15],'String','variable name','Tooltipstring','name of the variable','FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.4);

tool_text = uicontrol('Style','text','units','normalized','position',[0.1 0.62 0.3 0.15],'String','help string','Tooltipstring','help string','FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.4);

val_text = uicontrol('Style','text','units','normalized','position',[0.1 0.42 0.3 0.15],'String','value','Tooltipstring','value of the string','FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.4);

vis_flag = uicontrol('Style','checkbox','units','normalized','position',[0.35 0.22 0.3 0.15],'String','Visible','Tooltipstring','should the varialbe be visible???','FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.4,'Value',visible);

nam = uicontrol('Style','edit','units','normalized','position',[0.5 0.82 0.5 0.15],'String',name,'Tooltipstring','name of the variable','FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.3);

too = uicontrol('Style','edit','units','normalized','position',[0.5 0.62 0.5 0.15],'String',tool,'Tooltipstring','help string','FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.3);

str = get_text_string(value);

val= uicontrol('Style','edit','units','normalized','position',[0.5 0.42 0.5 0.15],'String',str,'Tooltipstring','value of the string','FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.3);

ask_for_fields = 0;



% wait for decision
while true;
  pause(0.1);
  drawnow
  if abs(ask_for_fields)==fig
    break;
  end
end

% evaluate decision and check if it makes sense
if ask_for_fields<0
  name = [];
else
  name = get(nam,'String');
  tool = get(too,'String');
  try
    eval(sprintf('value = %s;',get(val,'String')));
  catch
    set(f,'Visible','off');
    drawnow;
    message_box('Value entry not evaluable',1);
    name = [];
  end
  visible = get(vis_flag,'Value');
end

if isempty(name)
  tool = [];
  value = [];
  visible = [];
end

ask_for_fields = 0;

close(f);

