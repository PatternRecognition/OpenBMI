function [xfigure, yfigure]=axescoord2figurecoord(varargin)
% AXESCOORD2FIGURECOORD Transform axes coordinates in current
% figure units coordinate to the figure for annotation location
% [xfigure, yfigure]=axescoord2figurecoord(xaxes,yaxes)
% [xfigure, yfigure]=axescoord2figurecoord(xaxes,yaxes,handle_axes)
% 
% Ex.
%       % Create some data
% 		t = 0:.1:4*pi;
% 		s = sin(t);
%
%       % Add an annotation requiring (x,y) coordinate vectors
% 		plot(t,s);ylim([-1.2 1.2])
%       set(gcf,'Units','normalized');
% 		xa = [1.6 2]*pi;
% 		ya = [0 0];
% 		[xaf,yaf] = axescoord2figurecoord(xa,ya);
% 		annotation('arrow',xaf,yaf)
%
% Acknowledgments are due to Scott Hirsch (shirsch@mathworks.com) for is
% function ds2nfu. Some part of the present function derived from ds2nfu.
%
% Valley Benoît / Jan 2007
% valley@erdw.ethz.ch


% Process inputs
error(nargchk(2, 3, nargin))

if nargin==2
    xaxes=varargin{1};
    yaxes=varargin{2};
    h_axes = get(gcf,'CurrentAxes');
else
    xaxes=varargin{1};
    yaxes=varargin{2};
    h_axes = varargin{3};
end

% get axes properties
funit=get(get(h_axes,'Parent'),'Units');
% get axes properties
aunit=get(h_axes,'Units');
darm=get(h_axes,'DataAspectRatioMode');
pbarm=get(h_axes,'PlotBoxAspectRatioMode');
dar=get(h_axes,'DataAspectRatio');
pbar=get(h_axes,'PlotBoxAspectRatio');
xlm=get(h_axes,'XLimMode');
ylm=get(h_axes,'YLimMode');
xd=get(h_axes,'XDir');
yd=get(h_axes,'YDir');

% set the right units for h_axes
set(h_axes,'Units',funit);
axesoffsets = get(h_axes,'Position');

x_axislimits = get(h_axes, 'xlim');     %get axes extremeties.
y_axislimits = get(h_axes, 'ylim');     %get axes extremeties.
x_axislength = x_axislimits(2) - x_axislimits(1); %get axes length
y_axislength = y_axislimits(2) - y_axislimits(1); %get axes length

% mananged the aspect ratio problems


set(h_axes,'units','centimeters');
asc=get(h_axes,'Position');
rasc=asc(4)/asc(3);
rpb=pbar(2)/pbar(1);
if rasc<rpb
    xwb=axesoffsets(3)/rpb*rasc;
    xab=axesoffsets(1)+axesoffsets(3)/2-xwb/2;
    yab=axesoffsets(2);
    ywb=axesoffsets(4);
elseif rasc==rpb
    xab=axesoffsets(1);
    yab=axesoffsets(2);
    xwb=axesoffsets(3);
    ywb=axesoffsets(4);
else
    ywb=axesoffsets(4)*rpb/rasc;
    yab=axesoffsets(2)+axesoffsets(4)/2-ywb/2;
    xab=axesoffsets(1);
    xwb=axesoffsets(3);
end

if strcmp(darm,'auto') & strcmp(pbarm,'auto')
    xab=axesoffsets(1);
    yab=axesoffsets(2);
    xwb=axesoffsets(3);
    ywb=axesoffsets(4);
end

% compute coordinate taking in account for axes directions
if strcmp(xd , 'normal')==1
    xfigure = xab+xwb*(xaxes-x_axislimits(1))/x_axislength;
else
    xfigure = xab+xwb*(x_axislimits(2)-xaxes)/x_axislength;
end
if strcmp(funit,'normalized');
    xfigure(find(xfigure>1))=1;
    xfigure(find(xfigure<0))=0;
end


if strcmp(yd , 'normal')==1
    yfigure = yab+ywb*(yaxes-y_axislimits(1))/y_axislength;
else
    yfigure = yab+ywb*(y_axislimits(2)-yaxes)/y_axislength;
end
if strcmp(funit,'normalized');
    yfigure(find(yfigure>1))=1;
    yfigure(find(yfigure<0))=0;
end
set(h_axes,'Units',aunit); % put axes units back to original state