function mnt= adaptMontage(mnt, dat, appendix)
%mnt= adaptMontage(mnt, dat, appendix)

bbci_obsolete(mfilename, 'mnt_adaptMontage');

if ~exist('appendix','var'), appendix= ''; end


for ic= 1:length(mnt.clab),
  mnt.clab{ic}= [mnt.clab{ic} appendix];
end

avail= chanind(mnt, dat.clab);
mnt.x= mnt.x(avail);
mnt.y= mnt.y(avail);
mnt.clab= {mnt.clab{avail}};

avail_leg= [avail size(mnt.box,2)];  %% entry for legend
mnt.box= mnt.box(:,avail_leg);
mnt.box_sz= mnt.box_sz(:,avail_leg);
