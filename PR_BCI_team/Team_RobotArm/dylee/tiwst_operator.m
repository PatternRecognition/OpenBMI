
tic
%% Grasp (2 classes)
fprintf('Grasp')
load('grasp_1.mat');
    
    % Processing
fprintf("Processing")
for i = 1: 50
    fprintf(".")
    pause(0.2)
end

    
    % Result after cross validation = 1-error rate
    Result = 1 - C_eeg;
    Result_Std = loss_eeg_std;    
    % Cross-validation result 
    Result
    Result_Std
    

%% Twist (2 classes)
% Load pre-trained file
load('twist_pretained_4.mat');

% Find optimized performance based on pre-trained file
for sub= 1:length(filelist)
    % 최적의 성능 (퍼센트)
    performance(sub)=max(mean_acc(:,sub));
    com_idx=find(max(mean_acc(:,sub))==mean_acc(:,sub));
    f_idx=com(com_idx,:);
    % 최적의 주파수 대역
    max_frq(sub,:)=[frq_point(f_idx(1)),frq_point(f_idx(2))];
    
    clear com_idx f_idx
end

% Processing
fprintf("Processing")
for i = 1: 50
    fprintf(".")
    pause(0.2)
end

performance



toc