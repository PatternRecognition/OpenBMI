function mnt= mnt_shrinkNonEEGchans(mnt, nonEEGchans)
%mnt= mnt_shrinkNonEEGchans(mnt, <nonEEGchans>)

if ~exist('nonEEGchans', 'var'),
  nonEEGchans= chanind(mnt, 'E*');
else
  nonEEGchans= chanind(mnt, nonEEGchans);
end

mnt.box_sz(:,nonEEGchans)= 0.9*mnt.box_sz(:,nonEEGchans);
mnt.box(:,nonEEGchans)= mnt.box(:,nonEEGchans) + ...
    0.1*((sign(mnt.box(:,nonEEGchans))+1)/2);
