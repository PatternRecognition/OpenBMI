%%
excel_AUC = [];
ival_cfy_fixing = true;
EPO = cap_epo_sel; % cap_epo, cap_epo_sel
for j=1:10
%% select channels
chan = {'C3','C1','C2','C4','CP1','CP2','P3','Pz','P4','PO7','PO3','POz','PO4','PO8','O1','Oz','O2'};
% eval(sprintf('chan = %s_inter_chan{j};',modal));

%% ival setting
r_ival = [200 600];

% ival_cfy = [200 250; 250 300; 300 350; 350 400; 400 450];
ival_cfy = [200 250; 250 300; 300 350; 350 400; 400 450; 450 500; 500 550; 550 600; 600 650; 650 700; 700 750; 750 790];

%% training
epo = EPO{j,1};

% epo = proc_selectChannels(epo, chan);

% select ival
if ival_cfy_fixing == false
    epo_r= proc_selectIval(epo, r_ival, 'IvalPolicy','minimal');
    epo_r= proc_rSquareSigned(epo_r);
    ival_cfy= procutil_selectTimeIntervals(epo_r);
%     disp('ival cfy non-fixing  -  Check ival_cfy')
end

% slt_ival{j} = ival_cfy;

fv_Tr{j,1}= proc_jumpingMeans(epo, ival_cfy);

xsz= size(fv_Tr{j,1}.x);
fvsz= [prod(xsz(1:end-1)) xsz(end)];
% classifier C train_RLDAshrink
C = train_RLDAshrink(reshape(fv_Tr{j,1}.x,fvsz), fv_Tr{j,1}.y);

for speedIdx = 3:sum(~cellfun('isempty', cap_cnt(j,:)))
%% test
epo = EPO{j,speedIdx};

% epo = proc_selectChannels(epo, chan);

% fv_Te= proc_baseline(epo, ref_ival);
fv_Te{j,speedIdx-2}= proc_jumpingMeans(epo, ival_cfy);

xTesz= size(fv_Te{j,speedIdx-2}.x);
% test loss
outTe= apply_separatingHyperplane(C, reshape(fv_Te{j,speedIdx-2}.x, [prod(xTesz(1:end-1)) xTesz(end)]));
loss_all = loss_0_1(fv_Te{j,speedIdx-2}.y, outTe);
lossTe = mean(loss_all);
%         lossSem= std(lossTe,0,1)/sqrt(size(lossTe,1));

% training loss
outTr= apply_separatingHyperplane(C, reshape(fv_Tr{j,1}.x, fvsz));
% [loss_all,pred_results{j,speedIdx}, y_labels{j,speedIdx}] = loss_0_1(fv_Tr.y, outTr);
[loss_all] = loss_0_1(fv_Tr{j,1}.y, outTr);
lossTr = mean(loss_all);

excel_AUC(speedIdx-1, j) = 1-lossTe;

end
end

%%
disp('Mean AUC')
for ispeed = 1:4
mean_AUC(ispeed) = sum(excel_AUC(ispeed,:))/nnz(excel_AUC(ispeed,:));
end
disp(mean_AUC)

