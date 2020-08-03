function mrk= getBlinkMarkers_seg(cnt, chan, search_iv)
%mrk= getBlinkMarkers_seg(eeg, <chan, search_iv>)

if ~exist('chan', 'var'), chan='EOGv'; end

if ~exist('search_iv', 'var'),
  try
    marker= readMarkerTable(cnt.title);
  catch
    error('could not find marker table');
  end

  is= marker.pos(find(marker.toe==252));        %% start of segments
  ie= marker.pos(find(marker.toe==253));        %% end of segments
  if length(is)==0 | length(is)~=length(ie),
    error('no consistent segment information found');
  end

  ev= [];
  for ii= 1:length(is),
    mrk0= getBlinkMarkers_seg(cnt, chan, [is(ii) ie(ii)]/cnt.fs*1000);
    if ii==1,
      mrk= mrk0;
    else
      mrk.pos= [mrk.pos mrk0.pos];
    end
  end
  mrk.toe= ones(1, length(mrk.pos));
  mrk.y= mrk.toe;
  return
end


chan= chanind(cnt, chan);
iv= getIvalIndices(search_iv, cnt);
y= cnt.x(iv, chan);

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
bi= zeros(1, length(r));
for bl= 1:length(r),
  ival= round( r(bl) + [-0.3 0.3]*cnt.fs );
  [ma,mi]= max(abs(y(ival(1):ival(2))));
  bi(bl)= mi+ival(1)-1;
end

mrk.pos= bi + round(search_iv(1)*cnt.fs/1000);
mrk.toe= ones(1, length(mrk.pos));
mrk.fs= cnt.fs;
mrk.y= mrk.toe;
mrk.className= {'blink'};
