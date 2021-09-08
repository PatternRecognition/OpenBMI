clear all;
startup_bbci_toolbox

%% Data Load
dire = 'E:\ear_data\2020_paper_data\data_1906\SSVEP\data_publish\public_test_data';

for sub=1:10
    filename = sprintf('s%d',sub);
    epo_train{sub} = load(fullfile(dire,'train',filename)).epo;
    epo_test{sub} = load(fullfile(dire,'test',filename)).epo;
end


%% CCA based classifier setting
chan_cap = {'Oz'};
time_interval = [1002 2000];

freq = [11 7 5];  % [11 7 5] 5.45, 8.75, 12  [17 11 7 5]
fs = 500;
window_time = 1;
%(interval_sub(sub,2)-interval_sub(sub,1))/1000;

t = [1/fs:1/fs:window_time];

% ground truth
Y=cell(1);
for i=1:size(freq,2)
    Y{i}=[sin(2*pi*60/freq(i)*t);cos(2*pi*60/freq(i)*t);sin(2*pi*2*60/freq(i)*t);cos(2*pi*2*60/freq(i)*t)];
end

%% Training
for sub = 1:10

% channel select
epo = epo_train{sub};
epo = proc_selectChannels(epo, chan_cap);
epo = proc_selectIval(epo, time_interval);  % [0 4000]

% initialization
r_corr = []; r=[]; r_value=[]; pred=[];

% get features
nTrial = length(epo.y_dec);
for i=1:nTrial
    r_dump = [];
    for j=1:size(freq,2)
        [~,~, r_corr{j}] = canoncorr(squeeze(epo.x(:,:,i)),Y{j}');
        r_dump = [r_dump mean(r_corr{j})];
    end
    r(i,:) = r_dump;
    [r_value(i), pred(i)]=max(r(i,:));
end

% threshold
rere=r;
r_thres = rere.*(1-epo.y'); % non-target 만 남기기
thres_tr(sub,:) = mean(r_thres);

end

%% Test
excel_ACC = [];

for sub = 1:10
% channel select
epo = epo_test{sub};
epo = proc_selectChannels(epo, chan_cap);
epo = proc_selectIval(epo, time_interval);  % [0 4000]

% initialization
r_corr = []; r=[]; r_value=[]; pred=[];

% feature extraction
nTrial = length(epo.y_dec);
for i=1:nTrial
    r_dump = [];
    for j=1:size(freq,2)
        [~,~, r_corr{j}] = canoncorr(squeeze(epo.x(:,:,i)),Y{j}');
        r_dump = [r_dump mean(r_corr{j})];
    end
    r(i,:) = r_dump;
    % get threshold from training
    r2(i,:) = r(i,:)-thres_tr(sub,:);
    [r_value(i), pred(i)]=max(r2(i,:));
end
acc_all=length(find(epo.y_dec == pred))/nTrial;

excel_ACC(1,sub)=acc_all;
end

% accuracy
disp('Mean ACC')
mean_ACC = sum(excel_ACC(1,:))/nnz(excel_ACC(1,:));

disp(mean_ACC)

%% Get ITR
N = 3; % the number of class
P = excel_ACC;
duration = 1;
C = 60./duration';

ITR = (log2(N)+P.*log2(P)+(1-P).*log2((1-P)./(N-1))).*C;
% disp(ITR)
disp('Mean ITR')
mean(ITR)

