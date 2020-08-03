EEG_IMPORT_DIR = '/home/bci/eegImport/'

file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/graz/'];
file_list= {'O3VR', 'S4b', 'X11b'};


file= strcat(file_dir, file_list{su});
S= load(file);

cnt= struct('x',S.s, 'fs',S.HDR.SampleRate, ...
            'title',['IIIb/' file_list{su}]);
cnt.clab= {'C3','C4'};
mnt= setDisplayMontage(cnt.clab, 'C3,legend,C4');

%% substitute for NaN values.
%% stupid method. TODO: make it better
for cc= 1:size(cnt.x,2),
  iBad= find(isnan(cnt.x(:,cc)));
  for ii= 1:length(iBad),
    cnt.x(iBad(ii),cc)= cnt.x(iBad(ii)-1,cc);
  end
end

%% setup marker structures
mrk= struct('pos', S.HDR.TRIG', 'fs',cnt.fs);
mrk.y= zeros(3,length(mrk.pos));
for cc= 1:2,
  mrk.y(cc,:)= (S.HDR.Classlabel==cc);
end
mrk.y(3,:)= isnan(S.HDR.Classlabel);
mrk.className= {'left','right','test'};
clear S

mrk_left = mrk_selectClasses(mrk, 'left') ;
mrk_right = mrk_selectClasses(mrk, 'right') ;
mrk_test = mrk_selectClasses(mrk, 'test') ;

mrk_left_chron  = mrk_chronSplit(mrk_left, 4,{'t1','t2','t3','t4'}) ;
mrk_right_chron = mrk_chronSplit(mrk_right, 4,{'t1','t2','t3','t4'}) ;
mrk_test_chron  = mrk_chronSplit(mrk_test, 4,{'t1','t2','t3','t4'}) ;

mrk_lrt     = mrk_mergeMarkers(mrk_mergeMarkers(mrk_left, mrk_right),mrk_test);
mrk_lr_chron= mrk_mergeMarkers(mrk_left_chron, mrk_right_chron) ;
mrk_lr      = mrk_mergeMarkers(mrk_left, mrk_right) ;


%% spectra
figure;
set(gcf, 'name','Spectra L/R/T all trials') ;
epo= makeEpochs(cnt, mrk_lrt, [3500 7000]);
spec= proc_spectrum(epo, [2 30]);
grid_plot(spec, mnt,'shrinkAxes',[0.9, 1],'linewidth',2,'colorOrder',[ 1 0 0; 0 1 0 ;0 0 1 ]);
print(gcf, '-depsc2','-loose',sprintf('%sbroadSpec', file_list{su})) ;


figure;
set(gcf, 'name','Mu-Spectra L/R/T all trials') ;
spec= proc_spectrum(epo, [7 15]);
spec.clab= {'C3 \mu','C4 \mu'} ;
grid_plot(spec, mnt,'shrinkAxes',[0.9, 1],'linewidth',2,'colorOrder',[ 1 0 0; 0 1 0 ;0 0 1 ]);
print(gcf, '-depsc2','-loose',sprintf('%smuSpec', file_list{su})) ;

figure;
set(gcf, 'name','Beta-Spectra L/R/T all trials') ;
spec= proc_spectrum(epo, [15 26]);
spec.clab= {'C3 \beta','C4 \beta'} ;
grid_plot(spec, mnt,'shrinkAxes',[0.9, 1],'linewidth',2,'colorOrder',[ 1 0 0; 0 1 0 ;0 0 1 ]);
print(gcf, '-depsc2','-loose',sprintf('%sbetaSpec', file_list{su})) ;


figure;
set(gcf, 'name','Spectra L/R chronological') ;
epo= makeEpochs(cnt, mrk_lr_chron, [3500 7000]);
spec= proc_spectrum(epo, [2 30]);
grid_plot(spec, mnt,'shrinkAxes',[0.9, 1],'colorOrder',[ 1 .0 .20; 0.9 .0 .20 ;0.8 .10 .20 ; 0.7 .0 .20;.6 1 0;.6 .9 0; .6 .8 0; .6 .7 0],'linewidth',2);
print(gcf, '-depsc2','-loose',sprintf('%sbroadSpec_chron', file_list{su})) ;

figure;
set(gcf, 'name','mu-Spectra L/R chronological') ;
spec= proc_spectrum(epo, [7 15]);
grid_plot(spec, mnt,'shrinkAxes',[0.9, 1],'colorOrder',[ 1 .0 .20; 0.9 .0 .20 ;0.8 .10 .20 ; 0.7 .0 .20;.6 1 0;.6 .9 0; .6 .8 0; .6 .7 0],'linewidth',2);
print(gcf, '-depsc2','-loose',sprintf('%smuSpec_chron', file_list{su})) ;

figure;
set(gcf, 'name','beta Spectra L/R chronological') ;
spec= proc_spectrum(epo, [15 26]);
grid_plot(spec, mnt,'shrinkAxes',[0.9, 1],'colorOrder',[ 1 .0 .20; 0.9 .0 .20 ;0.8 .10 .20 ; 0.7 .0 .20;.6 1 0;.6 .9 0; .6 .8 0; .6 .7 0],'linewidth',2);
print(gcf, '-depsc2','-loose',sprintf('%sbetaSpec_chron', file_list{su})) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ERDs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[cnt_phase10Hz cnt_abs10Hz, wavelet_transformed10Hz] = phaseAmplitudeEstimate(cnt, 11) ;
epo= makeEpochs(cnt_abs10Hz, mrk_lrt, [500 7000]);
figure;
set(gcf, 'name','mu ERD L/R/T all trials') ;
grid_plot(epo,mnt,'shrinkAxes',[0.9, 1],'linewidth',2)
print(gcf, '-depsc2','-loose',sprintf('%smuERD', file_list{su})) ;

figure;
set(gcf, 'name','mu ERD L/R chronological') ;
epo= makeEpochs(cnt_abs10Hz, mrk_lr_chron, [500 7000]);
grid_plot(epo, mnt,'shrinkAxes',[0.9, 1],'colorOrder',[ 1 .0 .20; 0.9 .0 .20 ;0.8 .10 .20 ; 0.7 .0 .20;.6 1 0;.6 .9 0; .6 .8 0; .6 .7 0],'linewidth',2);
print(gcf, '-depsc2','-loose',sprintf('%smuERD_chron', file_list{su})) ;

epo= makeEpochs(cnt_abs10Hz, mrk_lr, [500 7000]);
fv= proc_selectIval(epo, [3500 7000]);
fv= proc_variance(fv, 5);
xvalidation(fv, 'LDA');



[cnt_phase20Hz cnt_abs20Hz, wavelet_transformed20Hz] = phaseAmplitudeEstimate(cnt, 22) ;
epo= makeEpochs(cnt_abs20Hz, mrk_lrt, [500 7000]);
figure;
set(gcf, 'name','beta Spectra L/R chronological') ;
grid_plot(epo,mnt,'shrinkAxes',[0.9, 1],'linewidth',2,'colorOrder',[ 1 0 0; 0 1 0 ;0 0 1 ])
print(gcf, '-depsc2','-loose',sprintf('%sbetaERD', file_list{su})) ;

figure;
set(gcf, 'name','beta ERD L/R chronological') ;
epo= makeEpochs(cnt_abs20Hz, mrk_lr_chron, [500 7000]);
grid_plot(epo, mnt,'shrinkAxes',[0.9, 1],'colorOrder',[ 1 .0 .20; 0.9 .0 .20 ;0.8 .10 .20 ; 0.7 .0 .20;.6 1 0;.6 .9 0; .6 .8 0; .6 .7 0],'linewidth',1);
print(gcf, '-depsc2','-loose',sprintf('%sbetaERD_chron', file_list{su})) ;

epo= makeEpochs(cnt_abs20Hz, mrk_lr, [500 7000]);
fv= proc_selectIval(epo, [5000 7000]);
fv= proc_variance(fv, 5);
xvalidation(fv, 'LDA');





%% slow potentials
figure
set(gcf,'name','ERP') ;
epo= makeEpochs(cnt, mrk_lrt, 3000+[0 2000]);
epo= proc_baseline(epo, 250, 'beginning');
grid_plot(epo, mnt,'shrinkAxes',[0.9, 1],'linewidth',2,'colorOrder',[ 1 0 0; 0 1 0 ;0 0 1 ]);
print(gcf, '-depsc2','-loose',sprintf('%sERP', file_list{su})) ;

figure
set(gcf,'name','subset ERP') ;
epo= makeEpochs(cnt, mrk_lr_chron, 3000+[0 2000]);
epo= proc_baseline(epo, 250, 'beginning');
epo= proc_selectIval(epo, 3000 + [500 1500]);
grid_plot(epo, mnt,'shrinkAxes',[0.9, 1],'colorOrder',[ 1 .0 .20; 0.9 .0 .20 ;0.8 .10 .20 ; 0.7 .0 .20;.6 1 0;.6 .9 0; .6 .8 0; .6 .7 0],'linewidth',2);
print(gcf, '-depsc2','-loose',sprintf('%sERP_chron', file_list{su})) ;

epo= makeEpochs(cnt, mrk_lr, 3000+[0 2000]);
epo= proc_baseline(epo, 250, 'beginning');
fv= proc_selectIval(epo, [3500 3750]);
fv= proc_jumpingMeans(fv, 4);
xvalidation(fv, 'LDA');
