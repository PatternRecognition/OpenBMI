function inter= metaintersect(zets, zet)
%inter= metaintersect(Z, z)
%
% intersect all sets of cell Z with set z

inter= cell(size(zets));
for iz= 1:length(zets),
  inter{iz}= intersect(zets{iz}, zet);
end
