opt_fig= strukt('folder', '/tmp/', 'format', 'pdf');

%% season 8
subdir= 'VPkg_08_11_19';                                     
file= [subdir '/imag_fbarrow_pmeanVPkg'];                

%% season 9
subdir= 'VPzq_09_01_28';
file= [subdir '/imag_fbarrow_pmeanVPzq'];

%% season 10
subdir= 'VPae_09_04_21';
file= [subdir '/relaxVPae'];

[mnt,hdr]= eegfile_loadMatlab(file, 'vars',{'mnt','hdr'})

ci= chanind(hdr, 'F7,8');
hdr.clab(ci)= {'F9','F10'};
mnt= getElectrodePositions(hdr.clab);

mnt= mnt_restrictMontage(mnt, {'not','E*'});

H= drawScalpOutline(mnt, 'showLabels',1);
set([H.nose H.head], 'LineWidth',2);
axis off
printFigure('electrode_layout_64mcc', [15 15], opt_fig);


%% project treder09
subdir= 'VPsag_09_03_13';
file= [subdir '/visual_p300_hex_targetVPsag'];

[mnt,hdr]= eegfile_loadMatlab(file, 'vars',{'mnt','hdr'})
mnt= getElectrodePositions(hdr.clab);
H= drawScalpOutline(mnt, 'showLabels',1);
set([H.nose H.head], 'LineWidth',2);
axis off
printFigure('electrode_layout_64std', [15 15], opt_fig);
