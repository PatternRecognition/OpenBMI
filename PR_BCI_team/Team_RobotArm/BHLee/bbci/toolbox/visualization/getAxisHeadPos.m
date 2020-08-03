function pos= getAxisHeadPos(mnt, ic, sz)
%pos= getAxisHeadPos(mnt, ic, <sz>)
%
% ic=0 or ic=nChans+1 for status box

if ic==0 | ic==length(mnt.clab)+1,
  ic= length(mnt.clab)+1;
  x= [mnt.x; max(mnt.x)];
  y= [mnt.y; max(mnt.y)];
else
  x= mnt.x;
  y= mnt.y;
end

bs= 0.005;
siz= (1+2*bs)*[max(x+sz(1)) - min(x) max(y+sz(2)) - min(y)];
pos= [x(ic)-min(x) y(ic)-min(y) sz(1) sz(2)]./[siz siz];
%pos= pos + [0 bs/2 0 0];
