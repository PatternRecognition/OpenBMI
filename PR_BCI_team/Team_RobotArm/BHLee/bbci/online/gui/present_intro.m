function fig = present_intro
% PRESENT AN INTRO!!!
%
% TODO: extended documentation by Schwaighase
% Guido Dornhege, 07/03/2005
% $Id: present_intro.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% OPENING THE FIGURE
%[im,cm] = bbci_logo;
[im,cm] = imread('bbci_logo.png');

fig = figure;   
scs = get(0,'ScreenSize');
set(fig,'Position',[0.5*(scs(3)-size(im,2)),0.5*(scs(4)-size(im,1)-200),size(im,2),size(im,1)+200]);
set(fig,'MenuBar','none');
set(fig,'NumberTitle','off');
set(fig,'Name','Introduction');
set(fig,'Units','pixel');
set(fig,'Color',[1 1 1]);

image(im);
colormap(cm);
axis off;
set(gca,'XLim',[1,size(im,2)]);
set(gca,'YLim',[1 size(im,1)+200]);

t = text(0.5*size(im,2),60+size(im,1),'STARTING BBCI-BET-INTERFACE (please wait)');
set(t,'HorizontalAlignment','center');
set(t,'FontSize',30);
t = text(0.5*size(im,2),120+size(im,1),'COPYRIGHT: THE BERLIN-BRAIN-COMPUTER-INTERFACE GROUP (Fraunhofer FIRST.IDA, UKBF)');
set(t,'HorizontalAlignment','center');
set(t,'FontSize',20);
t = text(0.5*size(im,2),170+size(im,1),'implemented by Anton Schwaighofer, Benjamin Blankertz, Guido Dornhege, Matthias Krauledat and Mikio Braun');
set(t,'HorizontalAlignment','center');
set(t,'FontSize',20);
pause(0.1);