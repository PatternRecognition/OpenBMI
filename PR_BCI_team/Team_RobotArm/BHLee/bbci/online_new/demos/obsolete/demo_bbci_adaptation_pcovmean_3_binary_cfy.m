% Subject-independent classifier
cfy_files= strcat(EEG_RAW_DIR, ...
                  'subject_independent_classifiers/vitalbci_season2/', ...
                  'Lap_C3z4_bp2_', ...
                  {'LR', 'LF', 'FR'});
% EEG file used of offline simulation of online processing
eeg_file= 'VPkg_08_08_07/imag_arrowVPkg';
[cnt, mrk_orig]= eegfile_loadMatlab(eeg_file, 'vars',{'cnt','mrk_orig'});

bbci= load(cfy_files{1});
bbci_cfy2= load(cfy_files{2}, 'classifier');
bbci_cfy3= load(cfy_files{3}, 'classifier');

bbci.source.acquire_fcn= @bbci_acquire_offline;
bbci.source.acquire_param= {cnt, mrk_orig};
bbci.source.marker_mapping_fcn= @marker_mapping_SposRneg;

% Set up three clasifiers: 1: 'L vs R';  2: 'L vs F';  3: 'F vs R'
bbci.classifier([2 3])= bbci.classifier(1);
bbci.classifier(2).C= bbci_cfy2.classifier.C;
bbci.classifier(3).C= bbci_cfy3.classifier.C;

bbci.control.classifier= [1 2 3];

bbci.adaptation.param{1}.ival= [750 4000];
bbci.adaptation.param{1}.mrk_end= [100];
bbci.adaptation([2 3])= bbci.adaptation(1);
bbci.adaptation(1).classifier= 1;
bbci.adaptation(1).param{1}.mrk_start= {1, 2};  % adapt L, R
bbci.adaptation(2).classifier= 2;
bbci.adaptation(2).param{1}.mrk_start= {1, 3};  % adapt L, F
bbci.adaptation(3).classifier= 3;
bbci.adaptation(3).param{1}.mrk_start= {3, 2};  % adapt F, R

% This is not really necessary: control signal could also be used.
bbci.log.classifier= 1;
bbci.log.file= fullfile(TMP_DIR, 'log');

data= bbci_apply(bbci);



% Evaluation: extract classifier outputs from log-file and
%   show traces of event-related classifier outputs for each combination

log_format= '%fs %s %s';
[time, cfystr, ctrlstr]= ...
    textread(data.log.filename, log_format, ...
             'delimiter','|','commentstyle','shell');
cfy= cell2mat(cellfun(@str2num, cfystr, 'UniformOutput',0));

cnt_cfy= struct('fs',25, 'x',[[0 0 0]; cfy], ...
                'clab',{{'cfy-LR','cfy-LF','cfy-FR'}});
mrk_cfy= mrk_orig;
mrk_cfy= mrk_defineClasses(mrk_orig, {1,2,3;'left','right','foot'});
mrk_cfy= mrk_resample(mrk_cfy, cnt_cfy.fs);
epo_cfy= cntToEpo(cnt_cfy, mrk_cfy, [0 5000]);
epo_cfy= proc_average(epo_cfy);
pairs= nchoosek(1:3,2);
colors= [1 0 1; 0 0.8 0; 0 0 1];
for pp= 1:size(pairs,1),
  fig_set(pp, 'gridsize',[2 3]); clf
  clidx= pairs(pp,1);
  plot(epo_cfy.t, epo_cfy.x(:,pp,clidx), 'Color',colors(clidx,:));
  hold on
  clidx= pairs(pp,2);
  plot(epo_cfy.t, epo_cfy.x(:,pp,clidx), 'Color',colors(clidx,:));
  xlabel('time post cue  [msec]');
  ylabel('classifier output  [a.u.]');
  legend(mrk_cfy.className(pairs(pp,:)));
  title(epo_cfy.clab{pp});
end
