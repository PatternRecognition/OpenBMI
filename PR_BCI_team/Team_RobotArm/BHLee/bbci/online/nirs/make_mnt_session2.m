function mnt=make_mnt_session2

clab.emit = {'Pz','C3','C1','FCz','C2','C4','F1','F2'};


clab.detect = {'POz','PCP1','PCP2', ...
	       'CCP3','CCP4', ...
	       'CFC5','CFC3','CFC1','CFC2','CFC4','CFC6',...
	      'F3','FAF1','Fz','FAF2','F4'};
 
[channel_info, mnt] = nirs_getChannelLayout(clab);
mnt.clabN=mnt.clab;
for i=1:length(channel_info)
mnt.clab{i}=channel_info{i}.EEGname;
end

mnt.x=1.7*mnt.x;
mnt.y=1.3*mnt.y;

