function mnt= restrictMontage(mnt, clab)
%mnt= restrictMontage(mnt, clab)

bbci_obsolete(mfilename, 'mnt_adaptMontage');

selectedChans= chanind(mnt, clab);
mnt.x= mnt.x(selectedChans);
mnt.y= mnt.y(selectedChans);
mnt.clab= mnt.clab(selectedChans);
mnt.box= mnt.box(:,selectedChans);
mnt.box_sz= mnt.box_sz(:,selectedChans);

if isfield(mnt, 'pos_3d'), mnt.pos_3d= mnt.pos_3d(selectedChans); end
