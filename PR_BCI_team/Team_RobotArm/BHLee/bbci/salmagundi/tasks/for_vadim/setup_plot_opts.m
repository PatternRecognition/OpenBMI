grid_opt.colorOrder= [1 0 0; 0 0.7 0; 0 0 1];
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOGh'}, {'EOGv'}};
grid_opt.scalePolicy= {'auto', [-10 10], 'sym', 'auto'};
grid_opt.axisTitleFontWeight= 'bold';

grd= sprintf('EOGh,F3,legend,F4,EOGv\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4\nEMGl,P3,EMGf,P4,EMGr');
mnt_lap= setDisplayMontage(mnt, grd);
mnt_lap= excenterNonEEGchans(mnt_lap, 'E*');
grd= sprintf('FC3,FC1,FCz,FC2,FC4\nC3,C1,Cz,C2,C4\nCP3,P3,legend,P4,CP4');
mnt_spec= setDisplayMontage(mnt, grd);
drag_down= chanind(mnt_spec, 'P3,4');
mnt_spec.box(2,drag_down)= mnt_spec.box(2,drag_down) - 0.13;
mnt_spec.box_sz= diag([0.9 0.8])*mnt_spec.box_sz;

spec_opt= grid_opt;
spec_opt.xTickMode= 'auto';
spec_opt.xTick= 10:5:30;
spec_opt.xTickLabelMode= 'auto';
spec_opt.scalePolicy= 'auto';

rsqu_opt= {'colorOrder',[0.9 0.9 0; 1 0 1; 0 0.8 0.8], ...
           'scalePolicy','auto'};

scalp_opt= struct('shading','flat', 'resolution',20, 'contour',-4);

colormap default;
