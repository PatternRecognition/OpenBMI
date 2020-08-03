function in = visualize_classifier(dat,mrk,pospi,mess);

persistent  info stat stop expert colbut linebut fhglogo range me si
global activation

if isempty(activation)
    activation = [1 1];
end


old = 0;


yellow = [1 1 0];
green = [0 1 0];
red= [1 0 0];
alphaf = 0.1;

if iscell(dat)
    


info  = figure;
set(info,'MenuBar','none','DoubleBuffer','on', 'Position', pospi);
stat = uicontrol('Style','listbox','Fontsize',6,'units','normalized','Position',[0.05,0.01,0.9,0.2]);
%set(stat,'Visible','off');

stop = uicontrol('Style','pushbutton','units','normalized','position',[0.82 0.75,0.17,0.1]);
set(stop,'Fontsize',6,'String','Quit');
expert = uicontrol('Style','pushbutton','units','normalized','position',[0.82 0.55,0.17,0.1]);
set(expert,'Fontsize',6,'String',sprintf('Restart'));
colbut = zeros(3,length(dat));
linebut = zeros(1,length(dat));
fhglogo = axes('units', 'normalized', 'Position', [0.82, 0.3, .17, .15]);

%[im,cm] = imread('fhg_first_logo.png');
[im,cm] = first_logo;
image(im); colormap(cm); axis equal;
axis off
for i = 1:length(dat)
    ttt = uicontrol('Style', 'text', 'String', dat{i}, ...
    'units', 'normalized', 'Fontsize',8,'Position', [0.05+(i-1)*0.8/length(dat),0.9,0.8/length(dat)-0.05,0.05]);

    h1 = axes('position',[0.05+(i-1)*0.8/length(dat),0.3,0.8/length(dat)-0.05,0.6]);
    hold on;
    colbut(1,i) = fill([0,1,1,0,0],[-3,-3,-1,-1,-3],green);
    colbut(2,i) = fill([0,1,1,0,0],[-3,-3,-1,-1,-3]+2,yellow);
    colbut(3,i) = fill([0,1,1,0,0],[-3,-3,-1,-1,-3]+4,red);
    linebut(i) = line([0,1],[0 0]);
    set(gca,'Xlim',[0,1]);
    set(gca,'Ylim',[-3,3]);
    axis off;
end
set(colbut,'FaceAlpha',alphaf);
set(linebut,'LineWidth',5,'Color',[0 0 1]);

copyright = uicontrol('Style', 'text', 'Fontsize',6,'String', 'Augcog Classifier v1.1 (c) 2004 Fraunhofer FIRST.IDA', ...
    'units', 'normalized', 'Position', [.0 .95 1 .05]);

if old
    range = mrk(:,2)-mrk(:,1);
    me = 0.5*(mrk(:,2)+mrk(:,1));
    si = 0.5*range;
    range = [mrk(:,1)-range, mrk(:,2)+range];
end

in = info;
return;
end



for i = 1:size(colbut,2)
    da = dat(i);
    if old
        da = da-me(i);
        da = da/si(i);
        da = sign(da)*log(1+abs(da));
    end
    set(linebut(i),'YData',[1 1]*min(3,max(-3,da)));
    set(colbut(1,i),'FaceAlpha',(dat(i+size(colbut,2))==0)*max(activation(i),alphaf)+(dat(i+size(colbut,2))==1)*alphaf);
    set(colbut(2,i),'FaceAlpha',alphaf);
    set(colbut(3,i),'FaceAlpha',(dat(i+size(colbut,2))==0)*alphaf+(dat(i+size(colbut,2))==1)*max(activation(i),alphaf));
end


if mess == 1
mr = cell(1,length(mrk));
for i = 1:length(mrk)
    mr{i} = sprintf('Marker: %d',mrk(i));
end

set(stat,'String',mr);
set(stat,'Value',length(mr));
end
