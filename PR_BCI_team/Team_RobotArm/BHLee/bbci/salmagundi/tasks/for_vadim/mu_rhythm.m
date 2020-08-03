sub_list= {'Guido_04_03_29/imag_lettGuido', 'Klaus_04_04_08/imag_lettKlaus'};
band= [9 11.5];

for ii= 1:length(sub_list),

file= strcat(sub_list{ii});
[cnt,mrk,mnt]= eegfile_loadMatlab(file);
mnt= mnt_setGrid(mnt, 'motor_cortex_ww7');
mnt= mnt_excenterNonEEGchans(mnt);

epo= makeEpochs(cnt, mrk, [-2000 2000]);
epo= proc_commonAverageReference(epo);
spec= proc_spectrum(epo, [5 45]);
spec= proc_classMean(spec);

figure; set(gcf, 'name',file);
grid_plot(spec, mnt, 'XTick',10:10:40);
grid_markIval(band);

figure; set(gcf, 'name',file);
scalpPattern(spec, mnt, band, 'colAx','range');

end




file= 'Vadim_05_08_23/imag_lettVadim';
cnt= readGenericEEG(file);
Mrk= readMarkerTable(file);
classDef= {1, 2, 3, 4; 'left','right','foot','relax'};
mrk= makeClassMarkers(Mrk, classDef);
mnt= getElectrodePositions(cnt.clab);
mnt= mnt_setGrid(mnt, 'motor_cortex_w7');

epo= makeEpochs(cnt, mrk, [-2000 2000]);
epo= proc_commonAverageReference(epo);
spec= proc_spectrum(epo, [5 45]);
spec= proc_classMean(spec);

figure; set(gcf, 'name',file);
grid_plot(spec, mnt, 'XTick',10:10:40);
grid_markIval(band);

figure; set(gcf, 'name',file);
scalpPattern(spec, mnt, band, 'colAx','range');





file= 'Vadim_05_08_23/selfpaced3sVadim';
cnt= readGenericEEG(file);
Mrk= readMarkerTable(file);
classDef= {-1, -2; 'left','right'};
mrk= makeClassMarkers(Mrk, classDef);

band= [9 11.5];
[b,a]= butter(5, band/(cnt.fs/2));
cnt_flt= proc_filtfilt(cnt, b, a);
epo= makeEpochs(cnt_flt, mrk, [-3000 1000]);
epo= proc_commonAverageReference(epo);
epo= proc_rectifyChannels(epo);
erd= proc_average(epo);
erd= proc_movingAverage(erd, 200);

grid_plot(erd, mnt);




file= 'Vadim_05_08_23/medianusVadim';
cnt= readGenericEEG(file);
Mrk= readMarkerTable(file);
classDef= {32; 'right'};
mrk= makeClassMarkers(Mrk, classDef);

band= [9 11.5];
[b,a]= butter(5, band/(cnt.fs/2));
cnt_flt= proc_filtfilt(cnt, b, a);
epo= makeEpochs(cnt_flt, mrk, [-2000 2000]);
epo= proc_commonAverageReference(epo);
epo= proc_rectifyChannels(epo);
erd= proc_average(epo);
erd= proc_movingAverage(erd, 200);

grid_plot(erd, mnt);
ival= [400 700];
grid_markIval(ival);

figure;
scalpPattern(erd, mnt, ival, 'colAx','range');
