fig_dir= 'preliminary/Hendrik_02_08_12/';

file= 'Hendrik_02_08_12/imagmultiHendrik'
[cnt, mrk, mnt]= loadProcessedEEG(file);

%% remove non-EEG channels (EMG, EOG)
cnt_eeg= proc_selectChannels(cnt, 'not','E*');
%cnt_eeg= proc_selectChannels(cnt, 'not','Fpz','AF3','AF4','TP7');
[b,a]= getButterFixedOrder([3 40], cnt_eeg.fs);
cnt_eeg= proc_filtfilt(cnt_eeg, b, a);

%% perform source separation on the first recording segment
tau= 0:5;
Mrk= readMarkerTable(file);
seg_start= Mrk.pos(min(find(Mrk.toe==252)));
seg_end= Mrk.pos(min(find(Mrk.toe==253)));
tic; W= tdsep0(cnt_eeg.x(seg_start:seg_end,:)', tau); toc


%% show scalp patterns of ICs
A= inv(W);
nICs= size(W,1);
opt.resolution= 32;
opt.shading= 'flat';
for ic= 1:nICs,
  suplot(nICs, ic, .01, .01);
  showScalpPattern(mnt, A(:,ic), 0, 'none', 'range', opt);
  hl= ylabel(sprintf('ic %d', ic)); set(hl,'visible','on');
end
%addTitle(cnt.title);
%saveFigure([fig_dir 'ica_patterns'], [10 6]*2);


pause


%% apply unmixing matrix to data
cnt_ic= copyStruct(cnt_eeg, 'x');
cnt_ic.x= cnt_eeg.x * W';
for ci= 1:length(cnt_ic.clab),
  cnt_ic.clab{ci}= sprintf('ic%d', ci);
end


%% show spectra of continuous ICs
chan_list= {'ic20','ic22','ic24','ic12'};
spec= proc_selectIval(cnt_ic, [seg_start seg_end]/Mrk.fs*1000);
spec= proc_spectrum(spec, [4 34]);
clf;
for ic= 1:length(chan_list),
  suplot(length(chan_list), ic, [0.08 0.12], [0.05]);
  showERP(spec, mnt, chan_list{ic});
%  xlabel('[Hz]');
  grid on;
end
shiftAxesUp;
%saveFigure([fig_dir 'ica_spec'], [10 6]*2);


%% show spectra of ICs in movement related epoches
classes= [1 2 3];  %% imagery of left/right hand resp foot movement
evt= find(any(mrk.y(classes,:)));
mrk.pos= mrk.pos(evt);
mrk.toe= mrk.toe(evt);
mrk.y= mrk.y(classes, evt);
mrk.className= {mrk.className{classes}};

epo= makeSegments(cnt_ic, mrk, [750 3250]);
spec= proc_spectrum(epo, [4 34], epo.fs);
chan_list= {'ic20','ic25','ic24','ic12','ic22'};
clf;
for ic= 1:length(chan_list),
  suplot(length(chan_list), ic, [0.1 0.12 0.08], [0.1 0.05 0.05]);
  showERP(spec, mnt, chan_list{ic});
%  xlabel('[Hz]');
  legend(epo.className);
  grid on;
end
unifyYLim;
%saveFigure([fig_dir 'ica_epo_spec'], [10 6]*2);

band= [8 12];
bandName= 'alpha';
[b,a]= getButterFixedOrder(band, cnt.fs);
cnt_flt= proc_filtfilt(cnt_ic, b, a);
cnt_flt= rmfield(cnt_flt, 'title');

refIval= [-500 0];
epo_flt= makeSegments(cnt_flt, mrk, [-500 4000]);
epo_flt= proc_squareChannels(epo_flt);
erd= proc_classMean(epo_flt, classes);
erd= proc_calcERD(erd, refIval, 100);

mnt_ic= setElectrodeMontage(cnt_ic.clab);
grd= sprintf('ic20,ic25,ic24\nic12,ic22,legend');
mnt_ic= setDisplayMontage(mnt_ic, grd);

showERPgrid(erd, mnt_ic);
%saveFigure([fig_dir 'ica_' bandName '_erd'], [10 6]*2);


band= [15 25];
bandName= 'beta';
[b,a]= getButterFixedOrder(band, cnt.fs);
cnt_flt= proc_filtfilt(cnt_ic, b, a);
cnt_flt= rmfield(cnt_flt, 'title');

refIval= [-500 0];
epo_flt= makeSegments(cnt_flt, mrk, [-500 4000]);
epo_flt= proc_squareChannels(epo_flt);
erd= proc_classMean(epo_flt, classes);
erd= proc_calcERD(erd, refIval, 100);

mnt_ic= setElectrodeMontage(cnt_ic.clab);
grd= sprintf('ic20,ic25,ic24\nic12,ic22,legend');
mnt_ic= setDisplayMontage(mnt_ic, grd);

showERPgrid(erd, mnt_ic);
%saveFigure([fig_dir 'ica_' bandName '_erd'], [10 6]*2);


opt.resolution= 40;
opt.contour= 1;
nChans= length(chan_list);
for ii= 1:nChans,
  ic= chanind(epo, chan_list{ii});
  suplot(nChans, ii, .05, .05);
  showScalpPattern(mnt, A(:,ic), 0, 'none', 'range', opt);
  hl= ylabel(sprintf('ic %d', ic)); set(hl,'visible','on');
end
%addTitle(cnt.title);
%saveFigure([fig_dir 'ica_patterns_selected'], [10 6]*2);
