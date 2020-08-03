function showImpedances(filedir, COL_AX, SHOW_LABELS, SCALE_POS)
% showImpedances(filedir, <COL_AX, SHOW_LABELS, SCALE_POS>)
%   or
% showImpedances({filebefore, fileafter}, ...)
%   or
% showImpedances({filebefore},...)
%
% IN:  filedir           directory containing impedance files
%                        'impedances_before' and 'impedances_after'
%        filebefore      truncated filename of 'before'-impedances 
%                        (i.e. without '_0_10.vhdr')
%        fileafter       truncated filename of 'after' -impedances
%      COL_AX            range for the colorbar. used for both subplots.
%      SHOW_LABELS       plot labels on electrode patches
%      SCALE_POS         in both subplots: position of the colorbar
%      
% recommendation: save the resulting figure with paperSize [35 20]
%                 (i.e. saveFigure('impedancefig', [35 20]))

% kraulem 10.09.2003
global EEG_RAW_DIR;


if ~exist('SHOW_LABELS','var')| isempty(SHOW_LABELS), SHOW_LABELS=1; end
if ~exist('SCALE_POS','var') | isempty(SCALE_POS), SCALE_POS='horiz'; end

if iscell(filedir),
  if length(filedir)>1
    filebefore= filedir{1};
    fileafter= filedir{2};
  else
    filebefore = filedir{1};
    fileafter = [];
  end
else
  if length(filedir)>1 & filedir(end)~='/',
    filedir= [filedir '/'];
  end
  filebefore= [filedir 'impedances_before'];
  fileafter= [filedir 'impedances_after'];
end

% load and show the "before"-impedances:
[lines, impedances] = readImpedances([EEG_RAW_DIR, filebefore]);
if ~exist('COL_AX','var') | isempty(COL_AX),
  zgMax= max(abs(impedances));
  COL_AX= [0 zgMax];
end

% cross out electrodes of the model which are not in use:
mnt = projectElectrodePositions;
[c, lines_index, mnt_index] = intersect(lines, mnt.clab);
mnt.x = mnt.x(mnt_index);
mnt.y = mnt.y(mnt_index);
mnt.pos_3d = mnt.pos_3d(:,mnt_index);
mnt.clab = {mnt.clab{mnt_index}};

lines = {lines{lines_index}};
impedances = impedances(lines_index);

clf;
if isempty(fileafter)
  plotImpedanceParts(impedances, mnt, 0, SHOW_LABELS, SCALE_POS, COL_AX);
  return
else
  plotImpedanceParts(impedances, mnt, 1, SHOW_LABELS, SCALE_POS, COL_AX);
end

% load and show the "after"-impedances:
[lines, impedances] = readImpedances([EEG_RAW_DIR, fileafter]);

% cross out electrodes of the model which are not in use:
mnt = projectElectrodePositions;
[c, lines_index, mnt_index] = intersect(lines, mnt.clab);
mnt.x = mnt.x(mnt_index);
mnt.y = mnt.y(mnt_index);
mnt.pos_3d = mnt.pos_3d(:,mnt_index);
mnt.clab = {mnt.clab{mnt_index}};

lines = {lines{lines_index}};
impedances = impedances(lines_index);

plotImpedanceParts(impedances, mnt, 2, SHOW_LABELS, SCALE_POS, COL_AX);

return




function fig = plotImpedanceParts(impedances, mnt, partNumber, ...
                                  SHOW_LABELS, SCALE_POS, COL_AX)
% plots the information from the array impedances into the subplot partNumber

%subplot(1,2,partNumber);
if partNumber>0
  subplot('Position', [(0.05+(partNumber-1)*.5) 0.05 0.4 0.9]);
end
dispChans= find(~isnan(mnt.x));
impedances= impedances(dispChans);

xe= mnt.x(dispChans);
ye= mnt.y(dispChans);

r= 0.05;
%inner circle:
T= linspace(0, 2*pi, 360);
xx= cos(T)/1.3;
yy= sin(T)/1.3;
plot(xx, yy, 'k');

T= linspace(0, 2*pi, 18);
disc_x= r*cos(T); 
disc_y= r*sin(T);

for ic= 1:length(dispChans),
  p =  patch(xe(ic)+disc_x, ye(ic)+disc_y, impedances(ic));
  %h= line(xe(ic)+disc_x, ye(ic)+disc_y);
  %set(h, 'color','k', 'lineWidth',.5);
  set(p, 'EdgeColor', 'none');
  set(p, 'FaceAlpha', 1);
end
caxis(COL_AX);
c = colormap(hsv(round((COL_AX(2)-COL_AX(1))*3.5)));
%brighten(c,.9);
c = c((COL_AX(2)-COL_AX(1)):(-1):1, :);
colormap(c);


hold on;
T= linspace(0, 2*pi, 360);
xx= cos(T);
yy= sin(T);
plot(xx, yy, 'k');
nose= [1 1.1 1];
nosi= [86 90 94]+1;
plot(nose.*xx(nosi), nose.*yy(nosi), 'k');

if SHOW_LABELS,
  labs= {mnt.clab{dispChans}};
  for il= 1:length(labs),
    strLen(il)= length(labs{il});
    labs{il}=[labs{il}, sprintf('\n%d', impedances(il))]; 
  end
  h= text(xe, ye, labs);
  set(h, 'fontSize',8, 'horizontalAlignment','center', 'verticalAlignment','middle');
  iLong= find(strLen>3);
  set(h(iLong), 'fontSize',7);
end

hold off;

if partNumber == 1
  ht= title('before');
  set(ht, 'fontSize',16);
elseif partNumber ==2
  ht= title('after');
  set(ht, 'fontSize',16);
end


axis('xy', 'tight', 'equal', 'tight','off');
if ~strcmp(SCALE_POS, 'none'),
  hb = colorbar(SCALE_POS);
  axes(hb);
  xlabel('[k\Omega]');
end


return