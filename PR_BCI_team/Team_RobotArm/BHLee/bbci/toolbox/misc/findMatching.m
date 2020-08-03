function [im, latencies]= findMatching(mrk, pos2, policy, max_latency)
%[im, latencies]= findMatching(mrk, mrk2/pos2, <policy, max_latency>)
%
% returns indices of best matching entries in mrk2.pos (resp pos2)
% to entries in mrk.pos.
% policy may be 'preceding', 'postceding', or (default) 'both'
% max_latency is given in msec.

if ~exist('policy','var') | isempty(policy), 
  policy= 'both'; 
end
if ~exist('max_latency','var'),
  max_latency= inf;
end
max_lat= max_latency/1000*mrk.fs;

if isstruct(pos2),
  if mrk.fs~=pos2.fs,
    error('inconsistent sampling rates');
  end
  pos2= pos2.pos;
end

im= [];
latencies= [];
for ei= 1:length(mrk.pos),
  switch(policy),
   case 'postceding',
    ik= min(find( pos2 >= mrk.pos(ei) ));
   case 'preceding',
    ik= max(find( pos2 <= mrk.pos(ei) ));
   otherwise,
    [lat, ik]= min(abs( pos2 - mrk.pos(ei) ));
  end
  lat= pos2(ik)-mrk.pos(ei);
  if ~isempty(ik) & abs(lat) < max_latency,
    im= [im, ik];
    latencies= [latencies, lat];
  end
end
latencies= latencies*1000/mrk.fs;

if length(unique(im))<length(im),
  warning('double assignments');
end
