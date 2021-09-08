clear all;
startup_bbci_toolbox

%% Load Isolated data

BTB.DataDir = 'E:\ear_data\2020_paper_data\data_1906';
BTB.paradigm = 'SSVEP';

BTB.MatDir = [BTB.DataDir '\' BTB.paradigm '\data_publish\' 'Mat_prep(ReSubNum)'];

switch(BTB.paradigm)
    case 'SSVEP'
        % Define some settings SSVEP
        disp_ival= [0 4000]; % SSVEP
        trig_all = {1,2,3, 11, 22; '5.45','8.57','12','Start','End'};
        trig_sti = {1,2,3; '5.45','8.57','12'};
%         subject = [1,2,3,5,6,8,11,12,15,16]; 
%         subject =  [1:6, 8,11,12,15];
%         subject = [1 2 3 5 6 8 10 11 12 15]; 
%         subject = [1,2,3,5,6,8,9,11,12,15]; %CCA
        subject = [1,2,3,5,6,8,10,11,12,15]; % accuracy
%         subject = 1:17;
    case {'vis_ERP', 'aud_ERP','vis_oddball','ERP','ERP_bp'}
        disp_ival= [-200 800]; % ERP
        ref_ival= [-200 0] ;
        trig_all = {2,1, 11, 22; 'target','non-target','Start','End'};
        trig_sti = {2,1 ;'target','non-target'};
        % 10Έν
        subject =  [2,4,6,7,8,10,11,16,17,18];
    case 'MI'
        disp_ival = [0 5000];
        trig_all = {1,2,3,11,22; 'right','left','rest','Start','End'};
        trig_sti = {1,2,3; 'right','left','rest'};
end

%%
subi = 0;
for subNum = subject 
subi = subi + 1;

%%
d_temp = load([BTB.MatDir '\' sprintf('s%d_prep.mat',subNum)]);
len = size(d_temp.cap_cnt,2);

cap_cnt(subi,1:len) = d_temp.cap_cnt;
cap_seg_mrk(subi,1:len) = d_temp.cap_seg_mrk;

cap_epo(subi,1:len) = d_temp.cap_epo_filt;
cap_mnt{subi,1} = d_temp.cap_mnt;

end

% %%
% for sub = 1:sum(~cellfun('isempty', cap_epo(:,1)))
% for ispeed = 1:sum(~cellfun('isempty', cap_epo(sub,:)))
% cap_epo{sub,ispeed} = proc_segmentation(cap_cnt{sub,ispeed},cap_seg_mrk{sub,ispeed},[0 6000]);
% end
% end

