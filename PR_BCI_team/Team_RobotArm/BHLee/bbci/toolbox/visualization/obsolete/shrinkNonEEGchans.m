function mnt= shrinkNonEEGchans(mnt, nonEEGchans)
%mnt= shrinkNonEEGchans(mnt, <nonEEGchans>)

bbci_obsolete(mfilename, 'mnt_shrinkNonEEGchans');

if ~exist('nonEEGchans', 'var'),
  nonEEGchans= chanind(mnt, 'E*');
else
  nonEEGchans= chanind(mnt, nonEEGchans);
end

mnt.box_sz(:,nonEEGchans)= 0.9*mnt.box_sz(:,nonEEGchans);
mnt.box(:,nonEEGchans)= mnt.box(:,nonEEGchans) + ...
    0.1*((sign(mnt.box(:,nonEEGchans))+1)/2);
