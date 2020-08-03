function ax= subplotxl(m, n, p, mv, mh)
%ax= subplotxl(m, n, p, mv, mh)
%
% m, n, p  are used like in subplot (but p can also be vector [x y]),
% mv, mh   specify the horizontal resp. vertical margin
%
% margins are 1 to 3 dimensional row vectors

if ~exist('mv','var') | isempty(mv), mv= 0; end
if ~exist('mh','var') | isempty(mh), mh= 0; end

if length(mv)==1, mv= [mv mv mv]; end
if length(mv)==2, mv= [mv mv(1)]; end
if length(mh)==1, mh= [mh mh mh]; end
if length(mh)==2, mh= [mh mh(1)]; end

if iscell(p),
  p1= p{1};
  p2= p{2};
  if length(p1)==1 & length(p2)>1,
    p1= p1*ones(1, length(p2));
  elseif length(p2)==1 & length(p1)>1,
    p2= p2*ones(1, length(p1));
  end
  ax= zeros(1, length(p1));
  for ii= 1:length(p1),
    ax(ii)= subplotxl(m, n, [p1(ii) p2(ii)], mv, mh);
  end
  return;
end

pv= ( 0.999 - mv(1) - mv(3) - mv(2)*(m-1) ) / (m-0.5);
ph= ( 0.999 - mh(1) - mh(3) - mh(2)*(n-1) ) / n;

if length(p)==1,
  iv= m - 1 - floor((p-1)/n);
  ih= mod(p-1, n);
else
  iv= m - p(1);
  ih= p(2) - 1;
end

pos= [mh(1) + ih*(mh(2)+ph),  mv(1) + iv*(mv(2)+pv)+0.04,  ph,  pv];

ax= axes('position', pos);

      