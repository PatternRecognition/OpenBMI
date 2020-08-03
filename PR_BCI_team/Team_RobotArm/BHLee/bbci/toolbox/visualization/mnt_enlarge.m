function mnt= mnt_enlarge(mnt, factor)

if nargin<2,
 factor= 0.98/max(sqrt(sum([mnt.x mnt.y].^2,2)));
end
mnt.x= factor * mnt.x;
mnt.y= factor * mnt.y;
