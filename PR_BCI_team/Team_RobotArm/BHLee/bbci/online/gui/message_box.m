function fig = message_box(str,flag)
%MESSAGE_BOX SHOWS A MESSAGE
% 
% usage:
%   h = message_box(str,flag);
%
% input:
%   str    string to show
%   flag   0: without ok button
%          1: with ok button
%
% output:
%   h      handle of the figure
%
% TODO: extended documentation by Schwaighase
% Guido Dornhege, 07/03/2005
% $Id: message_box.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

global message_box_status;
persistent show_box;

if isempty(show_box)
  show_box = false;
end

if isnumeric(str) | islogical(str)
  show_box = logical(str);
  return;
end

if show_box
  fig = figure;   
  scs = get(0,'ScreenSize');
  set(fig,'Position',[0.5*(scs(3)-600),0.5*(scs(4)-200),600,200]);
  set(fig,'MenuBar','none');
  set(fig,'NumberTitle','off');
  set(fig,'Name','Message');
  set(fig,'Units','Pixel');
  set(fig,'Color',[1 1 1]);
  
  t = text(300,100,untex(str));
  set(t,'FontSize',20);
  set(t,'HorizontalAlignment','center');
  axis off
  set(gca,'XLim',[0 600]);
  set(gca,'YLim',[0 200]);
  
  if flag==1
    b = uicontrol('Units','Pixel','Position',[300 10 90 40]);
    set(b,'Style','pushbutton');
    set(b,'Tooltipstring','Press to continue');
    set(b,'FontSize',20);
    set(b,'String','OK');
    set(b,'Callback','global message_box_status; message_box_status = false;');
    message_box_status = true;
    while message_box_status
      pause(0.2);
      drawnow;
    end
    close(fig);
  else
    pause(0.1);
    drawnow;
  end
else
  fprintf('%s\n',str);
end
