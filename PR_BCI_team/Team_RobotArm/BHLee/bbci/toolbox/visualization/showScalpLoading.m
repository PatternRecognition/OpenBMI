function H= showScalpLoading(mnt, w, SHOW_LABELS, SCALE_POS, COL_AX, CONTOUR, LABEL_SIZE, CIRCLE_LINE_WIDTH)
%showScalpLoading(mnt, w, <showLabels=1, scalePos, COL_AX, CONTOUR=0>)
%
% scalePos: 'horiz', 'vert', 'none'

% bb, GMD.FIRST.IDA, 08/00


if ~exist('SHOW_LABELS','var')| isempty(SHOW_LABELS), SHOW_LABELS=1; end
if ~exist('SCALE_POS','var') | isempty(SCALE_POS), SCALE_POS='vert'; end
if ~exist('COL_AX','var') | isempty(COL_AX),
  zgMax= max(abs(w));
  COL_AX= [-zgMax zgMax];
end
if ~exist('LABEL_SIZE','var')| isempty(LABEL_SIZE), LABEL_SIZE=8; end
if ~exist('CIRCLE_LINE_WIDTH','var')| isempty(CIRCLE_LINE_WIDTH), CIRCLE_LINE_WIDTH=1.5; end

cla;
dispChans= find(~isnan(mnt.x));
if length(w)==length(mnt.clab),
  w= w(dispChans);
else
  if length(w)~=length(dispChans),
    error(['length of w must match # of displayable channels, ' ...
           'i.e. ~isnan(mnt.x), in mnt']);
  end
end
xe= mnt.x(dispChans);
ye= mnt.y(dispChans);

%if eeg.nChans<=64,
%  r= 0.1;
%else
  r= 0.074;
%end
T= linspace(0, 2*pi, 18);
disc_x= r*cos(T); 
disc_y= r*sin(T);

labs= {mnt.clab{dispChans}};
strLen= apply_cellwise2(labs, inline('sum(isstrprop(x,''upper''))','x'));
iLong= find(strLen>=3);
%list= [dispChans(iLong); dispChans(find(strLen<3))]';
list= [iLong, find(strLen<3)];

for ic= list, %1:length(dispChans),
  H.patch(ic)= patch(xe(ic)+disc_x, ye(ic)+disc_y, w(ic));
  H.circle(ic)= line(xe(ic)+disc_x, ye(ic)+disc_y);
  set(H.circle(ic), 'Color','k', 'LineWidth', CIRCLE_LINE_WIDTH);
end
caxis(COL_AX);

hold on;
T= linspace(0, 2*pi, 360);
xx= cos(T);
yy= sin(T);
plot(xx, yy, 'k');
nose= [1 1.1 1];
nosi= [86 90 94]+1;
plot(nose.*xx(nosi), nose.*yy(nosi), 'k');

if SHOW_LABELS,
  H.labels= text(xe, ye, labs);
  set(H.labels, 'fontSize', LABEL_SIZE, 'horizontalAlignment','center');
  set(H.labels(iLong), 'fontSize', LABEL_SIZE-2);
end

hold off;
set(gca, 'xTick', [], 'yTick', []);
axis('xy', 'tight', 'equal', 'tight');
axis off;

if ~strcmp(SCALE_POS, 'none'),
  H.hcb= colorbarv6(SCALE_POS);
end

ii= find(isnan(w));
if ~isempty(ii),
  set(H.patch(ii), 'EdgeColor','none');
  set(H.labels(ii), 'Color',0.5*[1 1 1]);
  set(H.circle(ii), 'Color',0.5*[1 1 1], 'LineWidth', CIRCLE_LINE_WIDTH-0.5);
  moveObjectBack(H.labels(ii));
  moveObjectBack(H.circle(ii));
end

if nargout==0,
  clear H;
end
