function mnt= selectDisplayChannels(mnt, varargin);
%mnt= selectDisplayChannels(mnt, chans);
%
% IN   mnt   - display montage, see setElectrodeMontage, setDisplayMontage
%      chans - channels, format as accepted by chanind
%
% OUT  mnt   - updated display montage

chans= chanind(mnt.clab, varargin{:});
off= setdiff(1:length(mnt.clab), chans);

mnt.x(off)= NaN;
mnt.y(off)= NaN;
mnt.box(:,off)= NaN;
mnt.pos_3d(:,off)= NaN;
