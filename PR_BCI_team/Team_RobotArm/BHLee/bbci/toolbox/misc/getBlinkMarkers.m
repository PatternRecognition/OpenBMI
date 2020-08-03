function mrk= getBlinkMarkers(cnt, chan)
%mrk= getBlinkMarkers(eeg, <chan>)

if ~exist('chan', 'var'), chan='EOGv'; end

chan= chanind(cnt, chan);
y= cnt.x(:,chan);

scanLev= 5;
wfilt= 'bior6.8';
wLen= length(wfilters(wfilt));
[C,L]= wavedec(y, scanLev, wfilt);
v= C(L(1)+(1:L(2))).^2 / length(y);
t= length(v);

r= intersect(relMax(v), find(v>0.5*std(v)));
wHalfLen= wLen/2;
for i= 2:5,
  r= 2*r-wHalfLen;
end
r= 2*r-wHalfLen+1;
r = intersect(r,1:length(y));
bi= zeros(1, length(r));
for bl= 1:length(r),
  ival= round( r(bl) + [-0.3 0.3]*cnt.fs );
  [ma,mi]= max(abs(y(max(1,ival(1)):min(length(y),ival(2)))));
  bi(bl)= mi+ival(1)-1;
end

mrk.pos= bi;
mrk.toe= ones(1, length(mrk.pos));
mrk.fs= cnt.fs;
mrk.y= mrk.toe;
mrk.className= {'blink'};



return



try
  marker= readMarkerTable(cnt.title);
catch
  warning('could not find marker table');
  return;
end

is= marker.pos(find(marker.toe==252));        %% start of segments
ie= marker.pos(find(marker.toe==253));        %% end of segments
if length(is)==0 | length(is)~=length(ie),
  warning('no consistent segment information found');
  return;
end

ev= [];
for ii= 1:length(is),
  ev= [ev find(mrk.pos>is(ii) & mrk.pos<ie(ii))];
end
mrk= pickEvents(mrk, ev);
