function pos= getAxisGridPos(mnt, ic)
%pos= getAxisGridPos(mnt, ic)
%
% ic=0 or ic=nChans+1 for status box

nChans= length(mnt.clab);
if length(ic)==1 & ic==0,
  ic= nChans+1;
end

if ~isfield(mnt, 'box'),
  figure;
  ha= suplot(nChans+1, ic, .01, .01);
  pos= get(ha, 'position');
  close(gcf);
  return;
end

x= mnt.box(1,:);
y= mnt.box(2,:);
w= mnt.box_sz(1,:);
h= mnt.box_sz(2,:);
if isfield(mnt, 'scale_box'),
  x= cat(2, x, mnt.scale_box(1));
  y= cat(2, y, mnt.scale_box(2));
  w= cat(2, w, mnt.scale_box_sz(1));
  h= cat(2, h, mnt.scale_box_sz(2));
end  
if length(ic)==1,
  axpos= [x(ic) y(ic) w(ic) h(ic)];
else
  axpos= ic;
  if length(axpos)==2,
    axpos= [axpos 1 1];
  end
end

bs= 0.005;
siz= (1+2*bs)*[max(x+w) - min(x) max(y+h) - min(y)];
pos= [axpos(1)-min(x) axpos(2)-min(y) axpos([3 4])]./[siz siz];
pos= pos + [bs bs 0 0];
