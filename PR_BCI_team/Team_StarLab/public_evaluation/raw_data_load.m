%% Load Isolated data
% 
clc; clear all; close all;
startup_bbci_toolbox

BTB.DataDir = 'C:\Users\yelee\Desktop\test_tc2_4\raw_data';

preproc_flag = true; %false
%% pre-processing setting
band = [0.5];
fs=500;
[b,a]= butter(5, band/fs*2,'high');

%% subject setting
% sub = 5;
speedPool = {'stand', 'walk'};

for sub=1:10
% j= subNum;
% j=1;
%% changing setting
BTB.folderName = sprintf('s%d',sub); 

%% setting
BTB.RawDir= [BTB.DataDir '\' BTB.folderName];

disp_ival= [0 4000]; % SSVEP

trig_all = {1,2,3, 11, 22, 111,222; ...
    '5.45','8.57','12','Start','End','Rest start','Rest End'};
trig_sti = {1,2,3; '5.45','8.57','12'};

%% Data load
for ispeed = 1:2

filename = sprintf('s%d_%s',sub,speedPool{ispeed});

[cnt_orig, mrk_orig, hdr] = file_readBV(fullfile(BTB.RawDir,filename), 'Fs', 500);

% create mrk
mrk{sub,ispeed}= mrk_defineClasses(mrk_orig, trig_all);
mrk{sub,ispeed}.orig= mrk_orig;
seg_mrk{sub,ispeed} = mrk_defineClasses(mrk{sub,ispeed}, trig_sti);

if preproc_flag
    mnt= mnt_setElectrodePositions(cnt_orig.clab(1:32));

    % channel select
    cnt_ = proc_selectChannels(cnt_orig, 1:36);

    % bandpass
    cnt_ = proc_filtfilt(cnt_, b, a);

    cnt_.clab(33:36)=[];
    cnt_.clab{33} = 'EOGH';
    cnt_.clab{34} = 'EOGV';
    eog_x = cnt_.x(:,33:36);
    cnt_.x(:,33:36)=[];
    cnt_.x(:,33) = eog_x(:,1)- eog_x(:,2);
    cnt_.x(:,34) = eog_x(:,3)- eog_x(:,4);

    eeg = struct('data',cnt_.x',...
            'chanlocs', struct('labels',cnt_.clab),...
            'nbchan',length(cnt_.clab),'etc',[],...
            'srate',cnt_.fs);

    % RM EOG
    [eeg_RmO,State] = flt_eog('Signal', eeg, 'eogchans',{'EOGH','EOGV'},'removeeog',true);

    x = eeg_RmO.data';

    % RM line noise
    [eeg_RmL, ch_rm] = flt_clean_channels('Signal',eeg_RmO); % eeg_int eeg_RmO  %,'rereferenced',true

    if ~isempty([ch_rm.labels]) % 제거 된게 있다면 interpolation
        ch_rm_idx = find(ismember({eeg.chanlocs.labels}, {ch_rm.labels}));

        eeg_int = struct('data',x(:,1:32)',...
            'chanlocs', struct('labels',cnt_.clab(1:32),'X',num2cell(mnt.pos_3d(1,1:32)),'Y',num2cell(mnt.pos_3d(2,1:32)),'Z',num2cell(mnt.pos_3d(3,1:32)),'theta',[]),...
            'nbchan',length(cnt_.clab(1:32)),'srate',cnt_.fs,'xmax',max(max(abs(cnt_.x))),'xmin',0,...
            'etc',[],'trials',1,'epoch',[],'icaact',[],'specdata',[],'icachansind',[],'icasphere',[],'icawinv',[],'specicaact',[],'icaweights',[],...
            'event',[],'setname','');

        eeg_int = eeg_interp(eeg_int, eeg_int.chanlocs(ch_rm_idx), 'spherical');
        x = eeg_int.data';
    end

    % rereference
    x = bsxfun(@minus,x,mean(x,2));
    
    cnt_.x = x;
else
    cnt_ = cnt_orig;
end

cnt{sub,ispeed} = proc_selectChannels(cnt_, 1:32);


% segmentation
epo{sub,ispeed} = proc_segmentation(cnt{sub,ispeed}, seg_mrk{sub,ispeed}, disp_ival);

% nominal labeling
nTrial = length(seg_mrk{sub,ispeed}.time);
for i=1:nTrial
    epo{sub,ispeed}.y_dec(i) = find(epo{sub,ispeed}.y(:,i) == 1);
end

end
end

% create mnt
mnt= mnt_setElectrodePositions(cnt{1,1}.clab);

%% Train /  Test
epo_train = epo(:,1);
epo_test = epo(:,2);
