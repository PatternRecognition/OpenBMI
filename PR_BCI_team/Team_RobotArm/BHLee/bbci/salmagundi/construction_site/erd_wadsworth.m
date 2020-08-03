file= 'wadsworth/AA001';

[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [0 5500]);

epo_lap= proc_laplace(epo, 'large');
psd= proc_spectrum(epo_lap, [5 35], 64);

tiny_grid= sprintf('C3,Cz,C4\nCP3,legend,CP4');
mnt_tiny= setDisplayMontage(mnt, tiny_grid);
showERPgrid(psd, mnt_tiny); 
drawnow;

fprintf('press <ret> to calculate erd curves\n'); pause

band= [10 15]; 
%%band= [20 30];
refIval= [0 1500];
erd= proc_selectChannels(epo_lap, 'C3','Cz','C4','CP3','CP4');
[b,a]= getButterFilter(band, erd.fs);
erd= proc_filtfilt(erd, b, a);
erd= proc_squareChannels(erd);
erd= proc_classMean(erd, 1:4);
erd= proc_calcERD(erd, refIval, 100);

showERPgrid(erd, mnt_tiny, [-100 200]);
