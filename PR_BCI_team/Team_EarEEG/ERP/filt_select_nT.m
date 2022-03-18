
ref_ival = [-200 0];
new_ival = [200 790];


%%
for j=1:10
for i=1:sum(~cellfun('isempty', cap_cnt(j,:))) % speed
    
    idx_nT = find(cap_epo{j,i}.event.desc == 1);
    idx_T = find(cap_epo{j,i}.event.desc == 2);
    idx_rand = randperm(length(idx_nT));
    idx_nT_sel = idx_nT(idx_rand(1:length(idx_T)),1);

    idx_newTrial = sort([idx_T; idx_nT_sel]);


    cap_epo_sel{j,i} = proc_selectEpochs(cap_epo{j,i}, idx_newTrial);
    cap_epo_sel{j,i} = proc_baseline(cap_epo_sel{j,i}, ref_ival);
    cap_epo_sel{j,i} = proc_selectIval(cap_epo_sel{j,i}, new_ival);
    
end
end

