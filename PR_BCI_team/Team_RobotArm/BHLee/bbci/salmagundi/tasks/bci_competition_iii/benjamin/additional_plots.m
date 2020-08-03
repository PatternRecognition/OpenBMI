fig_dir= 'preliminary/bci_competition_iii/benjamin/';

[dmy,dmy,mnt]= ...
    loadProcessedEEG([EEG_IMPORT_DIR 'bci_competition_iii/data_set_ii_' su],...
                     '', 'mnt');

clf;
H= plotScalpPattern(mnt, zeros(1,64), 'showLabels',1, 'scalePos','none');
delete(H.patch);
saveFigure([fig_dir 'electrode_layout'], [12 12]);



file= [DATA_DIR 'siemensMat/Ben_05_01_07/einzelobjekteBen'];
[cnt,mrk,mnt]= loadProcessedEEG(file, 'display');

clab= 'Pz';

iValid= find(mrk.hit);
mk= mrk_selectEvents(mrk.trg, iValid);
epo= makeEpochs(cnt, mk, [-50 700]);
epo= proc_baseline(epo, [-50 50]);
epo= proc_selectClasses(epo, 'deviant');
erp= proc_average(epo);
ci= chanind(epo, clab);

clf;
h_ax= axes('position', [0.2 0.1 0.7 0.8]);
hold on;

dist= 40;
trials= [1:6];
nTrials= length(trials);
clear h_p h_l
for tt= 1:nTrials,
  h_p(tt)= plot(epo.t, epo.x(:,ci,trials(tt))-tt*dist);
  h_l(tt)= line(xlim, -tt*dist*[1 1]);
end

h_ave= plot(erp.t, erp.x(:,ci,1)-(nTrials+1.5)*dist);
set(h_ave, 'lineWidth',1.5);
h_l(nTrials+1)= line(xlim, -(nTrials+1.5)*dist*[1 1]);
set(h_l, 'lineStyle',':', 'color','k');

set(h_ax, 'YTick',-[nTrials+1.5 nTrials:-1:1]*dist, ...
          'YTickLabel',['ERP', sprintf('|trial %d', nTrials:-1:1)]);
setLastXTickLabel('[ms]');

h_dots= text(mean(xlim), -(nTrials+0.5)*dist, '...');
set(h_dots, 'fontSize',36, 'horizontalAli','center');

h_zl= line([0 0], ylim);
set(h_zl, 'color','k', 'lineStyle',':');
xx= 20;
yy= 15;
yLim= get(gca, 'YLim');
h_p= patch([-xx xx 0], yLim(1)+[0 0 yy], [1 0 0]);
h_p(2)= patch([-xx xx 0], yLim(2)+[0 0 -yy], [1 0 0]);
set(h_p, 'faceColor','r', 'edgeColor','none');

h_tit= title(clab);
set(h_tit, 'fontWeight','bold', 'fontSize',12);
saveFigure([fig_dir 'forming_erp'], [14 7]);



clear cnt epo erp mtk mnt


file= strcat('Guido_04_03_29/imag_', {'lett','move'}, 'Guido');
[cnt,mrk,mnt]= loadProcessedEEG(file);

ival= [1000 2000];
dist= 2.5;
tt= 1;

mrk= mrk_selectClasses(mrk, 'foot');
cnt= proc_laplace(cnt);
cnt= proc_selectChannels(cnt, 'C4');
[b,a]= butter(5, [9 13]/cnt.fs*2);
cnt_flt= proc_filt(cnt, b, a);

epo= makeEpochs(cnt, mrk, [0 4000]);
epo= proc_baseline(epo, ival(1) + [0 100]);
epo_flt= makeEpochs(cnt_flt, mrk, [0 4000]);
epo_rct= proc_rectifyChannels(epo_flt);
epo_ave= proc_average(epo_rct);
epo_smo= proc_movingAverage(epo_ave, 150);

iv= getIvalIndices(ival, epo);

clf;
hp= plot(epo.t(iv), -epo.x(iv,1,tt));
hold on;
hp(2)= plot(epo.t(iv), -epo_flt.x(iv,1,tt) + dist);
hp(3)= plot(epo.t(iv), -epo_rct.x(iv,1,tt) + 2*dist);
hp(4)= plot(epo.t(iv), -epo_ave.x(iv,1,tt) + 3.5*dist);
hp(5)= plot(epo.t(iv), -epo_smo.x(iv,1,tt) + 4.5*dist);

set(gca, 'XLim',[-50 50]+ival);
set(hp([4:end]), 'lineWidth',1.5);

h_dots= text(mean(xlim), 2.25*dist, '...');
set(h_dots, 'fontSize',36, 'horizontalAli','center');

posi= [0 1 2 3.5 4.5];
for pp= 1:length(hp),
  hl(pp)= line(xlim, [1 1]*posi(pp)*dist);
end
set(hl, 'lineStyle',':', 'color','k');
box off

set(gca, 'YDir','reverse', ...
         'YTick',posi*dist, 'YTickLabel', ...
         {'raw', 'band-pass', 'rectified', 'average', 'smoothing'});
yLim= get(gca, 'YLim');
yLim(2)= yLim(2) + dist/2;
set(gca, 'YLim',yLim);
setLastXTickLabel('[ms]');
saveFigure([fig_dir 'forming_erd'], [14 7]);



file= 'Gabriel_01_12_12/selfpaced2sGabriel';
[cnt,mrk,mnt]= loadProcessedEEG(file, 'display');

mnt= projectElectrodePositions(cnt.clab);
grd= sprintf('EOGh,legend,Fz,scale,EOGv\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4\nEMGl,P3,Oz,P4,EMGr');
mnt= mnt_setGrid(mnt, grd);
grid_opt= struct('colorOrder',[0 0.7 0; 0 1 0.7]);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOGh'}, {'EOGv'}};
grid_opt.scalePolicy= {'auto', 'auto', 'sym', 'auto'};
grid_opt= set_defaults(grid_opt, ...
                       'lineWidth',1, 'axisTitleFontWeight','bold', ...
                       'axisType','cross', 'visible','off', ...
                       'scale_hpos', 'left', ...
                       'figure_color',[1 1 1]);
scalp_opt= struct('shading','flat', 'resolution',50, ...
                  'colAx','sym', 'colormap',jet(20), ...
                  'contour',-5, 'contour_policy','strict');

%iLeft= find(mrk.y(1,:));
%iRight= find(mrk.y(2,:));
%nlh= floor(length(iLeft)/2);
%nrh= floor(length(iRight)/2);
%N= length(iLeft)+length(iRight);
%mrk.y= [ismember(1:N, iLeft(1:nlh)); ...
%        ismember(1:N, iLeft(nlh+1:end)); ...
%        ismember(1:N, iRight(1:nrh)); ...
%        ismember(1:N, iRight(nrh+1:end))];
%mrk.className= {'early left','late left', 'early right', 'late right'};

mrk= mrk_selectClasses(mrk, 'right');
nEh= floor(size(mrk.y,2)/2);
mrk.y= [[1;0]*ones(1,nEh), [0;1]*ones(1,size(mrk.y,2)-nEh)];
mrk.className= {'early right', 'late right'};

epo= makeEpochs(cnt, mrk, [-1200 600]);
epo= proc_baseline(epo, [-1200 -1000]);
epo_rsq= proc_r_square(epo);

H= grid_plot(epo, mnt, grid_opt);
grid_addBars(epo_rsq, 'h_scale',H.scale);
saveFigure([fig_pre 'lrp_nonstat'], [19 12]);



file= 'Guido_02_01_08/selfpaced2sGuido';
[cnt,mrk,mnt]= loadProcessedEEG(file, 'display');

mnt= projectElectrodePositions(cnt.clab);
grd= sprintf('EOGh,legend,FCz,scale,EOGv\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');
mnt= mnt_setGrid(mnt, grd);

[b,a]= butter(5, [8 13]/cnt.fs*2);
cnt_flt= proc_laplace(cnt);
cnt_flt= proc_filt(cnt_flt, b, a);

mrk= mrk_selectClasses(mrk, 'right');
nEh= floor(size(mrk.y,2)/2);
mrk.y= [[1;0]*ones(1,nEh), [0;1]*ones(1,size(mrk.y,2)-nEh)];
mrk.className= {'early right', 'late right'};

epo= makeEpochs(cnt_flt, mrk, [-1500 500]);
epo= proc_rectifyChannels(epo);
epo= proc_movingAverage(epo, 150);
epo= proc_baseline(epo, [-1500 -1250]);
epo_rsq= proc_r_square(epo);
epo= proc_average(epo);

H= grid_plot(epo, mnt, grid_opt);
grid_addBars(epo_rsq, 'h_scale',H.scale);
saveFigure([fig_pre 'erd_nonstat'], [19 12]);



file= [DATA_DIR 'siemensMat/Ben_05_01_07/kofferBen'];
[cnt,mrk,mnt]= loadProcessedEEG(file, 'display');

mnt= projectElectrodePositions(cnt.clab);
grd= sprintf('AF7,legend,Fz,scale,AF8\nTP7,P3,Pz,P4,TP8\nP7,PO3,POz,PO4,P8');
mnt= mnt_setGrid(mnt, grd);

iValid= find(mrk.hit);
mrk= mrk_selectEvents(mrk.trg, iValid);
iStd= find(mrk.y(1,:));
n= floor(length(iStd)/3);
mrk= mrk_setClasses(mrk, {iStd(1:n), iStd(end-n+1:end)}, ...
                    {'early std', 'late std'});

epo= makeEpochs(cnt, mrk, [-850 150]);
spec= proc_spectrum(epo, [5 35]);

H= grid_plot(spec, mnt, 'colorOrder',grid_opt.colorOrder);
saveFigure([fig_dir 'spec_nonstat'], [19 12]);



file= 'Guido_04_03_18/imag_lettGuido';
[cnt,mrk,mnt]= loadProcessedEEG(file);

epo= makeEpochs(cnt, mrk, [750 4000]);
spec= proc_spectrum(epo, [5 35], epo.fs);
spec= proc_selectClasses(spec, 'left','foot');

plotClassTopographies(spec, mnt, [8 15], scalp_opt, ...
                      'colAx','range', 'show_title',0);
saveFigure([fig_dir 'naive_spectrum'], [18 10]);

clf;
showERP(spec, mnt, 'C3');
xlabel('Frequency [Hz]'); ylabel('Spectral Power [dB]');
shiftAxesUp;
saveFigure([fig_dir 'spectrum_C3_naive'], [8 6]);
clf;
showERP(spec, mnt, 'C4');
xlabel('Frequency [Hz]'); ylabel('Spectral Power [dB]');
shiftAxesUp;
saveFigure([fig_dir 'spectrum_C4_naive'], [8 6]);
clf;
showERP(spec, mnt, 'Cz');
xlabel('Frequency [Hz]'); ylabel('Spectral Power [dB]');
shiftAxesUp;
saveFigure([fig_dir 'spectrum_Cz_naive'], [8 6]);

spec= proc_spectrum(epo, [5 35], epo.fs);
spec_ref= proc_classMean(spec);
spec_ref.className= {'mean'};
spec= proc_selectClasses(spec, 'left','foot');
spec= proc_subtractReferenceClass(spec, spec_ref);

plotClassTopographies(spec, mnt, [8 15], scalp_opt, ...
                      'colAx','range', 'show_title',0);
saveFigure([fig_dir 'spectrum_relative_to_baseline'], [18 10]);


epo= proc_laplace(epo);
spec= proc_spectrum(epo, [5 35], epo.fs);
%spec_ref= proc_classMean(spec);
spec= proc_selectClasses(spec, 'left','foot');
%spec= proc_subtractReferenceClass(spec, spec_ref);

clf;
showERP(spec, mnt, 'C3');
xlabel('Frequency [Hz]'); ylabel('Spectral Power [dB]');
shiftAxesUp;
saveFigure([fig_dir 'spectrum_C3_lap'], [8 6]);
clf;
showERP(spec, mnt, 'C4');
xlabel('Frequency [Hz]'); ylabel('Spectral Power [dB]');
shiftAxesUp;
saveFigure([fig_dir 'spectrum_C4_lap'], [8 6]);
clf;
showERP(spec, mnt, 'Cz');
xlabel('Frequency [Hz]'); ylabel('Spectral Power [dB]');
shiftAxesUp;
saveFigure([fig_dir 'spectrum_Cz_lap'], [8 6]);
