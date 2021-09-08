%% CCA based classifier

chan_cap = {'Oz'};
chan = {'PO3','POz','PO4','O1','Oz','O2'};
% chan = {'Oz','POz'};

excel_AUC = [];

nSub=sum(~cellfun('isempty', cap_epo(:,1)));

time_interval = [1502 2000];

%%
freq = [11 7 5];  % [11 7 5] 5.45, 8.75, 12  [17 11 7 5]
fs = 500;
window_time = 0.5;
%(interval_sub(sub,2)-interval_sub(sub,1))/1000;

t = [1/fs:1/fs:window_time];

% ground truth
Y=cell(1);
for i=1:size(freq,2)
    Y{i}=[sin(2*pi*60/freq(i)*t);cos(2*pi*60/freq(i)*t);sin(2*pi*2*60/freq(i)*t);cos(2*pi*2*60/freq(i)*t)];
end

%%
for sub = 1:sum(~cellfun('isempty', cap_epo(:,1)))

%train = false;  % true: ispeed=1, false: ispeed=3
%% Train
for ispeed = 1%sum(~cellfun('isempty', cap_epo(sub,:)))
%% channel select
epo = cap_epo{sub,ispeed};
epo = proc_selectChannels(epo, chan_cap);
epo = proc_selectIval(epo, time_interval);  % [0 4000]

%% accuracy
% initialization
r_corr = []; r=[]; r_value=[]; pred=[];

for i=1:nTrial
    r_dump = [];
    for j=1:size(freq,2)
        [~,~, r_corr{j}] = canoncorr(squeeze(epo.x(:,:,i)),Y{j}');
        r_dump = [r_dump mean(r_corr{j})];
    end
    r(i,:) = r_dump;
    [r_value(i), pred(i)]=max(r(i,:));
end
acc_all=length(find(epo.y_dec == pred))/nTrial;
rere=r;
r_thres = rere.*(1-epo.y'); % non-target 만 남기기
thres_tr(sub,:) = mean(r_thres);

% excel_AUC(ispeed,sub)=acc_all;
end

%% test
for ispeed = 2%sum(~cellfun('isempty', cap_epo(sub,:)))
%% channel select
epo = cap_epo{sub,ispeed};
epo = proc_selectChannels(epo, chan_cap);
epo = proc_selectIval(epo, interval_sub(sub,:));  % [0 4000]

%% accuracy
% initialization
r_corr = []; r=[]; r_value=[]; pred=[];

for i=1:nTrial
    r_dump = [];
    for j=1:size(freq,2)
        [~,~, r_corr{j}] = canoncorr(squeeze(epo.x(:,:,i)),Y{j}');
        r_dump = [r_dump mean(r_corr{j})];
    end
    r(i,:) = r_dump;
    r2(i,:) = r(i,:)-thres_tr(sub,:);
    [r_value(i), pred(i)]=max(r2(i,:));
end
acc_all=length(find(epo.y_dec == pred))/nTrial;

excel_AUC(ispeed,sub)=acc_all;
end
end
%%
disp('Mean AUC')
for ispeed = 1:2
mean_AUC(ispeed) = sum(excel_AUC(ispeed,:))/nnz(excel_AUC(ispeed,:));
end
disp(mean_AUC)

