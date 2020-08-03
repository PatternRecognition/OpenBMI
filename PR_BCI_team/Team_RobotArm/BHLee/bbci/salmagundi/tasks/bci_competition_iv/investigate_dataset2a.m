DATDIR= '/home/blanker/data/bci_competition_iv/dataset_2a/';

cd('~/matlab/Import');
biosig_installer

file_list= cprintf('A%02dT', 1:9);


vp= 0;
vp= vp+1

filename= [DATA_DIR 'eegMat/bci_competition_iv/' file_list{vp}];
[cnt, mrk, mnt]= eegfile_loadMatlab(filename);

epo= cntToEpo(cnt, mrk, [-500 2000]);
epo= proc_baseline(epo, 250);
%grid_plot(epo, mnt, defopt_erps);

epo2= proc_selectClasses(epo, 'left','right');
%epo2= proc_selectClasses(epo, 'foot','tongue');
epor= proc_r_square_signed(epo2);
H= grid_plot(epo2, mnt, defopt_erps);
grid_addBars(epor, 'h_scale',H.scale, ...
             'cLim','sym','colormap',cmap_posneg(21));




return

vp= 0;

vp= vp+1,

filename= [DATA_DIR 'eegMat/bci_competition_iv/' file_list{vp}];
[cnt, mrk, mnt]= eegfile_loadMatlab(filename);

ii= find(isnan(cnt.x(:)));
cnt.x(ii)= 100000 * sign(mod(1:length(ii),2)-0.5);
reject_varEventsAndChannels(cnt, mrk, [-500 4000], 'visualize',1, 'whiskerlength',1.5);
