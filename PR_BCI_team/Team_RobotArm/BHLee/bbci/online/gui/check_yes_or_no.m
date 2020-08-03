function flag = check_yes_or_no(string);

global check_yes_or_no_status;

check_yes_or_no_status  = [];

fig = figure;   
scs = get(0,'ScreenSize');
set(fig,'Position',[0.5*(scs(3)-600),0.5*(scs(4)-200),600,200]);
set(fig,'MenuBar','none');
set(fig,'NumberTitle','off');
set(fig,'Name','Message');
set(fig,'Units','Pixel');
set(fig,'Color',[1 1 1]);
set(fig,'DeleteFcn','closereq; global check_yes_or_no_status; check_yes_or_no_status = false;');

t = text(300,100,untex(string));
set(t,'FontSize',20);
set(t,'HorizontalAlignment','center');
axis off
set(gca,'XLim',[0 600]);
set(gca,'YLim',[0 200]);


b = uicontrol('Units','Pixel','Position',[450 10 100 40]);
set(b,'Style','pushbutton');
set(b,'Tooltipstring','Yes');
set(b,'FontSize',20);
set(b,'String','Yes');
set(b,'Callback','global check_yes_or_no_status; check_yes_or_no_status = true;');

b2 = uicontrol('Units','Pixel','Position',[50 10 100 40]);
set(b2,'Style','pushbutton');
set(b2,'Tooltipstring','No');
set(b2,'FontSize',20);
set(b2,'String','No');
set(b2,'Callback','global check_yes_or_no_status; check_yes_or_no_status = false;');

while isempty(check_yes_or_no_status)
  pause(0.2);
  drawnow;
end

flag = check_yes_or_no_status;
close(fig);
