% function MapMontage (Montage)
% 
% This routine maps the 2-D locations of an EEG montage.
%
% Usage: MapMontage (Montage);
%
%   Input argument:  Montage   cell structure returned by the CSD toolbox 
%                              ExtractEEGMontage.m consisting of a channel
%                              label 'lab', 2-D plane x-y coordinates 'xy',
%                              and 3-D spherical angles 'theta' and 'phi'
%        
% Copyright (C) 2009 by Jürgen Kayser (Email: kayserj@pi.cpmc.columbia.edu)
% GNU General Public License (http://www.gnu.org/licenses/gpl.txt)
% Updated: $Date: 2009/05/14 14:10:00 $ $Author: jk $
%
function MapMontage (Montage)
if nargin < 1
  disp('*** Error: No EEG montage specified');
  return
end
nElec = size(Montage.xy,1);
set(gcf,'Name',sprintf('%d-channel EEG Montage',nElec),'NumberTitle','off')
m = 100;
t = [0:pi/100:2*pi]; 
r = m/2 + 0.5;
head = [sin(t)*r + m/2+1; cos(t)*r + m/2+1]' - m/2;
scrsz = get(0,'ScreenSize');
d = min(scrsz(3:4)) / 2;
set(gcf,'Position',[scrsz(3)/2 - d/2 scrsz(4)/2 - d/2 d d]); 
whitebg('w');
axes('position',[0 0 1 1]);
set(gca,'Visible','off');
line(head(:,1),head(:,2),'Color','k','LineWidth',1); 
mark = '\bullet'; 
if nElec > 35; mark = '.'; end;  
for e = 1:nElec
    text(Montage.xy(e,1)*m - m/2,Montage.xy(e,2)*m - m/2,mark);
    text(Montage.xy(e,1)*m - m/2 + 1*m/100, ...
         Montage.xy(e,2)*m - m/2 - 4*m/100, ...
         Montage.lab(e), ...
          'FontSize',8, ...
          'FontWeight','bold', ...
          'VerticalAlignment','middle', ...
          'HorizontalAlignment','center');
end
