function mnt= restrictDisplayChannels(mnt, varargin);
%mnt= restrictDisplayChannels(mnt, chans);
%
% IN   mnt   - display montage, see setElectrodeMontage, setDisplayMontage
%      chans - channels, format as accepted by chanind
%
% OUT  mnt   - updated display montage

bbci_obsolete(mfilename, 'mnt_adaptMontage');

chans= chanind(mnt.clab, varargin{:});
%off= setdiff(1:length(mnt.clab), chans);

mnt.x = mnt.x(chans);
mnt.y = mnt.y(chans);
mnt.box = mnt.box(:,chans);
mnt.box_sz = mnt.box_sz(:,chans);
mnt.clab= mnt.clab(chans);

if isfield(mnt, 'pos_3d'),
  mnt.pos_3d = mnt.pos_3d(:,chans);
end
