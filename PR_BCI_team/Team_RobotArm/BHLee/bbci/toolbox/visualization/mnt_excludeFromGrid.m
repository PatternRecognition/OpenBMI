function mnt= mnt_excludeFromGrid(mnt, varargin)

off= chanind(mnt.clab, varargin{:});
mnt.box(:,off)= NaN;
mnt.box_sz(:,off)= NaN;
