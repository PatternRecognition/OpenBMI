file= strcat('Klaus_04_04_08/imag_', {'lett','move'}, 'Klaus');
[cnt,mrk,mnt]= loadProcessedEEG(file, 'display');

band= [8 26];
[b,a]= butter(5, band/cnt.fs*2);

cnt_flt= proc_filtfilt(cnt, b, a);

epo= makeEpochs(cnt_flt, mrk, [-500 4000]);
epo= proc_rectifyChannels(epo);
epo= proc_movingAverage(epo, 200, 'centered');
epo= proc_baseline(epo, [-500 0]);
erd= proc_average(epo);
figure(1); set(gcf, 'name','plain ERDs');
scalpEvolutionPlusChannel(erd, mnt, 'CP2', 1000:750:4000, 'legend_pos',2);

erd_ref= proc_classMean(erd);
erd_ref.className= {'mean'};
erd= proc_subtractReferenceClass(erd, erd_ref);
figure(2); set(gcf, 'name','ERDs with global ERD subtracted');
scalpEvolutionPlusChannel(erd, mnt, 'CP2', 1000:750:4000, 'legend_pos',2);

epo= makeEpochs(cnt_flt, mrk, [-500 4000]);
epo= proc_laplace(epo);
epo= proc_rectifyChannels(epo);
epo= proc_movingAverage(epo, 200, 'centered');
epo= proc_baseline(epo, [-500 0]);
erd= proc_average(epo);
figure(3); set(gcf, 'name','ERDs from laplace filtered channels');
scalpEvolutionPlusChannel(erd, mnt, 'CP2', 1000:750:4000, 'legend_pos',2);
