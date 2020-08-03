subdir_list= textread([BCI_DIR 'studies/season3/session_list'], '%s');

grd= sprintf(['scale,F3,Fz,F4,legend\n' ...
              'C3,C1,Cz,C2,C4\n' ...
              'CP3,CP1,CPz,CP2,CP4\n' ...
              'P5,P3,Pz,P4,P6']);


for vp= 1%:length(subdir_list),

subdir= [subdir_list{vp} '/']
is= min(find(subdir=='_'));
sbj= subdir(1:is-1);
sd= subdir;
sd(find(ismember(sd,'_/')))= [];

%% default settings
ival_erd= [-500 4500];
classes= {'left','right'};

%% load settings that have been used for classification, if available
file= [subdir 'imag_lett' sbj];
S= who('-FILE', [EEG_MAT_DIR file '_cut50.mat']);
if isempty(strmatch('bbci',S)),
  fprintf('no feedback performed.\n');
else
  bbci= eegfile_loadMatlab([file '_cut50'],'vars','bbci');
  classes= bbci.classes;
end

[cnt, mrk]= eegfile_loadMatlab(strcat(file, '_cut50'));
[mrk, rClab]= reject_varEventsAndChannels(cnt, mrk, ival_erd);
mrk= mrk_selectClasses(mrk, classes);

mnt= setElectrodeMontage(cnt.clab);
mnt= mnt_setGrid(mnt, grd);

%% comment-out, if you do not want to use Laplace
disp_clab= getClabOfGrid(mnt);
requ_clab= getClabForLaplace(cnt, disp_clab);
cnt= proc_selectChannels(cnt, requ_clab);
cnt= proc_laplace(cnt);

epotfr= makeEpochs(cnt, mrk, ival_erd);
epotfr= proc_tfr(epotfr);
er= proc_r_square_signed(epotfr);

figure(1)
grid_image(proc_selectClasses(epotfr,1), mnt, 'climmode','range');
figure(2)
grid_image(proc_selectClasses(epotfr,2), mnt, 'climmode','range');
figure(3)
grid_image(er, mnt, 'climmode','sym');

end
