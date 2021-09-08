%%
ival_cfy_fixing = 1; % 1:ture // fixing

excel_AUC = [];
excel_SNR = [];
excel_SNR_db = [];
slt_ival = [];

for subNum=1:15
%% select channels
% erp_chan=15:28;
chan = {'C3','C1','C2','C4','CP1','CP2','P3','Pz','P4','PO7','PO3','POz','PO4','PO8','O1','Oz','O2'};
% eval(sprintf('chan = %s_inter_chan{j};',modal));

%% ival setting

ref_ival= [-200 0] ;
r_ival = [100 600];
% psn_ival = [-150 -50] ;
psn_ival = [-200 0] ;
p300_ival = [280 380];

% ival_cfy = [200 250; 250 300; 300 350; 350 400; 400 450];
ival_cfy = [100 110; 110 120; 120 130; 130 140; 140 150; ...
    150 160; 160 170; 170 180; 180 190; 190 200; ...
    200 210; 210 220; 220 230; 230 240; 240 250; ...
    250 260; 260 270; 270 280; 280 290; 290 300; ...
    300 310; 310 320; 320 330; 330 340; 340 350; ...
    350 360; 360 370; 370 380; 380 390; 390 400;...
    400 410; 410 420; 420 430; 430 440; 440 450;...
    450 460; 460 470; 470 480; 480 490; 490 500];


%% training
epo = epo_train{subNum};
epo = proc_selectChannels(epo, chan);

% select ival
if ival_cfy_fixing == false
epo_r= proc_selectIval(epo, r_ival, 'IvalPolicy','minimal');
epo_r= proc_rSquareSigned(epo_r);
ival_cfy= procutil_selectTimeIntervals(epo_r);
disp('ival cfy non-fixing  -  Check ival_cfy')
end

slt_ival{subNum} = ival_cfy;

fv_Tr{subNum} = proc_baseline(epo, ref_ival);
fv_Tr{subNum} = proc_jumpingMeans(fv_Tr{subNum}, ival_cfy);

xsz= size(fv_Tr{subNum}.x);
fvsz= [prod(xsz(1:end-1)) xsz(end)];

% classifier C train_RLDAshrink
C = train_RLDAshrink(reshape(fv_Tr{subNum}.x,fvsz), fv_Tr{subNum}.y);

%% test
epo = epo_test{subNum};

epo = proc_selectChannels(epo, chan);

fv_Te{subNum}= proc_baseline(epo, ref_ival);
fv_Te{subNum}= proc_jumpingMeans(fv_Te{subNum}, ival_cfy);

xTesz= size(fv_Te{subNum}.x);
% test loss
outTe= apply_separatingHyperplane(C, reshape(fv_Te{subNum}.x, [prod(xTesz(1:end-1)) xTesz(end)]));
lossTe = mean(loss_0_1(fv_Te{subNum}.y, outTe));

% training loss
outTr= apply_separatingHyperplane(C, reshape(fv_Tr{subNum}.x, fvsz));
[loss_all] = loss_0_1(fv_Tr{subNum}.y, outTr);
lossTr = mean(loss_all);

pred_prop(:,subNum) = outTe;
epo.y_dec = epo.y(1,:);
[ERP_per.roc, ERP_per.auc]= roc_curve(epo.y, outTe,'plot',0);

excel_AUC(subNum) = ERP_per.auc;


end
%%
disp('Mean AUC')
mean_AUC = sum(excel_AUC)/nnz(excel_AUC);

disp(mean_AUC)

