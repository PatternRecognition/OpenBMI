file= 'Steven_01_11_20/selfpaced5sSteven';

[cnt, mrk, mnt]= loadProcessedEEG(file);
mvt= makeSegments(cnt, mrk, [-600 400]);

mvt_psd= proc_laplace(mvt, 'small', '');
mvt_psd= proc_selectChannels(mvt_psd, 'C3','C4');
mvt_psd= proc_spectrum(mvt_psd, [4 40], 64);
ER_mvt= proc_classMean(mvt_psd, 1:2);

%mnt_psd= setDisplayMontage(mnt, 'C3C4');
showERP(ER_mvt, mnt);


band= [15 30];

[b,a]= getButterFilter(band, cnt.fs);
cnt_flt= proc_filtfilt(cnt, b, a);
cnt_flt.title= sprintf('%s [%d %d] Hz', cnt.title, band);
cnt_lap= proc_laplace(cnt_flt, 'small', ' lap', 'CP3','CP4');
cnt_lap= proc_selectChannels(cnt_lap, 'C3','Cz','C4','CP3','CP4');

refIval= [-3500 -2500];
epo= makeSegments(cnt_lap, mrk, [-3500 4000]);
%epo= proc_baseline(epo, refIval);
epo= proc_squareChannels(epo);
erd= proc_classMean(epo, 1:2);
erd= proc_calcERD(erd, refIval, 100);

mnt_c= setDisplayMontage(mnt, sprintf('C3,Cz,C4\nCP3,legend,CP4'));
showERPgrid(erd, mnt_c);







return

Filter_list= {'alpha2','beta','alpha2beta2'};
 
filter= Filter_list{2};
[cnt, mrk, mnt]= loadProcessedEEG(file, filter);
cnt.title= [cnt.title ' [' filter ']'];
cnt_lap= proc_laplace(cnt, 'small', ' lap', 'CP3','CP4');
cnt_lap= proc_selectChannels(cnt_lap, 'C3','Cz','C4','CP3','CP4');
 
refIval= [-3500 -2500];
epo= makeSegments(cnt_lap, mrk, [-3500 4000]);
%epo= proc_baseline(epo, refIval);
epo= proc_squareChannels(epo);
erd= proc_classMean(epo, 1:2);
erd= proc_calcERD(erd, refIval, 100);
 
mnt_c= setDisplayMontage(mnt, sprintf('C3,Cz,C4\nCP3,legend,CP4'));
showERPgrid(erd, mnt_c);
