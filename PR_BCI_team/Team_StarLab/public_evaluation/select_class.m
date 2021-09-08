% selection
%% Make FFT features
for sub = 1:sum(~cellfun('isempty', cap_epo(:,1)))

%train = false;  % true: ispeed=1, false: ispeed=3
for ispeed = 1:sum(~cellfun('isempty', cap_epo(sub,:)))

% cap_epo_sel{sub,ispeed} = proc_selectClasses(cap_epo{sub,ispeed},[1,2,3]);
%% y_dec
nTrial = size(cap_epo{sub,ispeed}.y,2);
cap_epo{sub,ispeed}.y_dec=ones(1,nTrial);
for i=1:nTrial
    cap_epo{sub,ispeed}.y_dec(i) = 2-find(cap_epo{sub,ispeed}.y(:,i) == 1);
end
end
end

%% save
dire = 'E:\ear_data\2020_paper_data\data_1906\SSVEP\data_publish\public_test_data_acc';

for sub = 1:sum(~cellfun('isempty', cap_epo(:,1)))

%train = false;  % true: ispeed=1, false: ispeed=3
for ispeed = 2
epo = cap_epo{sub,ispeed};

filename = sprintf('s%d',sub);
save(fullfile(dire,'test',filename), 'epo');

disp('Saved')
end
end

