su = 1;

global EEG_RAW_DIR EEG_MAT_DIR DATA_DIR

EEG_MAT_DIR = '/cdrom/';

bad_clab = {};bad_trials = [];

switch su
 case 1
  subject= 'Ben'; date_str= '05_01_07';
  thresh = [1.5,2];
  bad_clab = {'TP8'};
 case 2
  subject= 'Frank'; date_str= '05_01_11';
  thresh = [2,3];
 case 3
  subject= 'Thorsten'; date_str = '05_02_22';
  thresh = [1,1.5];
 case 4
  subject= 'Markus'; date_str = '05_03_21';
  thresh = [1,1.2];
end


sub_dir= [subject '_' date_str '/'];
fig_dir= ['siemens_paper/'];

grid_opt= struct('colorOrder',[0 0 0; 0.4,0.4,0.4;0.7,0.7,0.7]);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOGh'}, {'EOGv'}};
grid_opt.scalePolicy= {'auto', 'auto', 'sym', 'auto'};
grid_opt= set_defaults(grid_opt, ...
                       'lineWidth',1, 'axisTitleFontWeight','bold', ...
                       'axisType','cross', 'visible','off', ...
                       'figure_color',[1 1 1]);
spec_opt= grid_opt;
spec_opt.yUnit= 'power';
spec_opt.scalePolicy= 'auto';

scalp_opt= struct('shading','interp', 'resolution',20, 'contour',-4);

band= [8 12; 30 45];
band_name= {'mu', 'hyperbeta'};
nBands= length(band_name);


wind = 1000;
shift = -1200;

trials1 = [400:550,600:678];
trials2 = [1390:1450,1590:1645,1787:1806];
trials3 = [1310:1388,1510:1585,1657:1785];

string = sprintf('EOGh,F3,F1,Fz,F2,F4,EOGv\nFC5,FC3,FC1,FCz,FC2,FC4,FC6\nC5,C3,C1,Cz,C2,C4,C6\nCP5,CP3,CP1,CPz,CP2,CP4,CP6\nEMGl,P3,scale,Pz,legend,P4,EMGr');

exp_name = 'koffer';


file_name= strcat(sub_dir, exp_name, subject);
fprintf('load the data for %s\n',file_name);
%[cnt,mrk,mnt] = loadProcessedEEG(file_name,'display');
[cnt,mrk,mnt] = eegfile_loadMatlab(file_name);
cnt = proc_selectChannels(cnt,'not',bad_clab{:});
fprintf('laplace filtering\n');
cnt= proc_laplace(cnt, 'small','');
mrk = mrk_selectEvents(mrk,setdiff(1:size(mrk.y,2),bad_trials));

mrk.pos = mrk.pos([trials1,trials2,trials3]);
mrk.toe = [ones(1,length(trials1)),2*ones(1,length(trials2)),3*ones(1,length(trials3))];
mrk = rmfields(mrk,{'hit','trg','pic_list','pic','indexedByEpochs','lat'});
mrk.y = [mrk.toe==1;mrk.toe==2;mrk.toe==3];
mrk.className = {'Low CI','Med CI','High CI'};

mrk = mrk_sortChronologically(mrk);

fprintf('windowing\n');
epo_lap= makeEpochs(cnt, mrk, [-300 wind]+shift);
clear cnt mrk
%% !! baseline überflüssig für spec
fprintf('baseline\n');
epo_lap= proc_baseline(epo_lap, [-300 0]+shift);

fprintf('select ival\n');
spec_lap= proc_selectIval(epo_lap, [0 wind]+shift);
clear epo_lap;
fprintf('calculate spectrum\n');
spec= proc_spectrum(spec_lap, [5 45], spec_lap.fs);
clear spec_lap;

mn = setDisplayMontage(mnt,sprintf('F3,Fz,F4\nC3,Cz,C4\nCP3,CPz,CP4\nP3,Pz,P4\nscale,Oz,legend'));
%mn = setDisplayMontage(mnt,sprintf('C3,Cz,C4\nCP3,CPz,CP4\nscale,Pz,legend'));
fprintf('plot\n\n');
spec.title = sprintf('Subject: %d, Spectra 5-45 Hz',su);
hh = gray(64); hh = hh(end:-1:1,:); hh(1,:) = [0.999999,0.999999,0.999999];colormap(hh);

H = grid_plot(spec, mn, spec_opt,'titleDir','none');
spec_rsqu= proc_r_square_signed(proc_selectClasses(spec,{'Low CI','High CI'}));
grid_addBars(spec_rsqu, 'h_scale',H.scale,'colormap',hh);
addTitle(spec.title);

keyboard

saveFigure([fig_dir subject '_' exp_name '_conc_lap_spec_selectedtrials'], [10 10]);

spec_rsqu= proc_r_square_signed(spec);

clf;
for ib= 1:nBands,
  for ic = 1:length(spec_rsqu.className)
  subplot(length(spec_rsqu.className),nBands,nBands*(ic-1)+ib);
  topo= proc_meanAcrossTime(proc_selectClasses(spec_rsqu,ic), band(ib,:));
  %plotScalpPattern(setElectrodeMontage(spec_rsqu.clab), topo.x(:,:,1), scalp_opt);
  scalpPlot(setElectrodeMontage(spec_rsqu.clab), topo.x(:,:,1), scalp_opt);
  title(sprintf('%s  [%d-%d Hz]', topo.className{1}, band(ib,:)));
  end
end
addTitle(untex(topo.title), 1, 0);
saveFigure([fig_dir subject '_' exp_name '_conc_spec_rsqu_topo_selectedtrials'], [10 10]);