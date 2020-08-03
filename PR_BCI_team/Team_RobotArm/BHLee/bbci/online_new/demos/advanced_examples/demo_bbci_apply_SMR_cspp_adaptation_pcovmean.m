global TMP_DIR

% TMP_DIR= [DATA_DIR 'results/vitalbci_season1/csp_patches/online_new/'];
TMP_DIR= [EEG_RAW_DIR 'VPjj_11_07_18_tmp/'];

d= pwd;
d= d(find(d=='/',1,'last')+1:end);

% EEG file used of offline simulation of online processing
subdir= 'VPjj_08_06_10';
sbj = subdir(1:find(subdir=='_',1,'first')-1);
eeg_file= [subdir '/imag_fbarrow' sbj];

[cnt, mrk_orig, mrk]= eegfile_loadMatlab(eeg_file, 'vars',{'cnt','mrk_orig', 'mrk'});

patch_centers= {'C3','Cz','C4'};
patch_centers_str= 'C3z4';
patch = 'small';
clab= getClabForLaplacian(cnt, patch_centers, 'filter_type', patch);

band = [8 32];
bandstr= strrep(sprintf('%g-%g', band'),'.','_');
[filt_b, filt_a]= butters(5, band/cnt.fs*2);

clstag = [upper(mrk.className{1}(1)), upper(mrk.className{2}(1))];
cfy_name= ['patches_' patch_centers_str '_' patch '_' bandstr '_' clstag '.mat'];
cfy_name= [EEG_RAW_DIR '/subject_independent_classifiers/season13/' cfy_name];
bbci= load(cfy_name);

%740 is just to match with the 750 of the old version. Usually we put 750
%here.
bbci.feature.ival= [-740 0];

bbci.source.acquire_fcn= @bbci_acquire_offline;
bbci.source.acquire_param= {cnt, mrk_orig};
bbci.source.marker_mapping_fcn= @marker_mapping_SposRneg;

bbci.quit_condition.marker= 254;

bbci.log.output= 'screen&file';
bbci.log.file= [TMP_DIR '/log_cspp_pcovmean'];
bbci.log.classifier= 1;
bbci.log.force_overwriting= 1;

uc = 2.^(-[10:5:80]./8);
iUC_mean= 2;
iUC_pcov= 3;
bbci.adaptation.active= 1;
bbci.adaptation.fcn= @bbci_adaptation_pcovmean;
bbci.adaptation.param= {struct('ival',[750 3750],'UC_mean', uc(iUC_mean),'UC_pcov', uc(iUC_pcov), 'mrk_start', {{1, 2}}, 'mrk_end', [11, 12, 21, 22])};
bbci.adaptation.filename= '$TMP_DIR/bbci_classifier_pcovmean';

bbci.control.fcn= @bbci_control_SMR_cursor_1d;
bbci.control.param= {struct('mrk60',1, 'mrk_start', [1 2], 'mrk_end', [11 12 21 22])};
bbci.control.condition.marker= [1,2,11,12,21,22,60];

if isequal(d, 'demos')
   % local bbci_apply_evalCondition. Needed to calculate the results.
   % Otherwise no control is needed
  bbci.control.condition.interval= 40;
end

data= bbci_apply(bbci);

%% extract simulated online classification results
% read markers and classifier output from logfile
if isequal(d, 'demos')
  % local bbci_apply_evalCondition used. Needed to calculate the results
  results= textread(data.log.filename, '%s');
  markers= results(3:9:end);
  idx_resp = [];
  for m= 1:length(bbci.control.condition.marker(3:end-1))    
    idx_resp= [idx_resp strmatch(['M(' num2str(bbci.control.condition.marker(3+m-1)) ')'], markers)'];
  end
  control= results(idx_resp*9);
  for trial= 1:length(idx_resp)
    idx1 = findstr(control{trial},',')+1;
    idx2 = findstr(control{trial},']')-1;
    ishit(trial)= str2num(control{trial}(idx1(end):idx2));
  end
  disp([int2str(sum(ishit)) '%'])
else
  % svn bbci_apply_evalCondition used
  [time, marker_desc, marker_time, cfy_output, control]= textread(data.log.filename, log_format, ...
    'delimiter','','commentstyle','shell');
  
  % map mrk_orig.desc strings to intergers
  nEvents= length(time);
  [mrk_desc, iSR]= marker_mapping_SposRneg(mrk_orig.desc);
  % and pick those, which evoked the calculation of classifier outputs
  idx= find(ismember(mrk_desc, bbci.control.condition.marker));
  idx_control= idx(1:nEvents);
  
  % validate makers that evoked calculation of control signals
  isequal(marker_desc', mrk_desc(idx_control))
end
