function mnt= excenterNonEEGchans(mnt, nonEEGchans)
%mnt= excenterNonEEGchans(mnt, <nonEEGchans>)

bbci_obsolete(mfilename, 'mnt_excenterNonEEGchans');

if ~exist('nonEEGchans', 'var'),
  nonEEGchans= chanind(mnt, 'E*');
else
  nonEEGchans= chanind(mnt, nonEEGchans);
end

mnt.box(:,nonEEGchans)= mnt.box(:,nonEEGchans) + ...
    0.1*sign(mnt.box(:,nonEEGchans));
