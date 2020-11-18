%% Brain connectivity
% connectivity_weighted phase leg index
clc; clear;close all;
format;

srate=250;

addpath('C:\Users\Doyeunlee\Desktop\Analysis\eeglab14_1_2b\');
eeglab;

%% frequency bands

% range = {[0.5 4],[4 8],[8 13],[13 30],[30 50]};
range = {[4 8],[8 13],[13 40]}; % theta, mu,beta band
% range = {[0.5 4]}; %Hz
% emo={[1,2,3,5,6,7,10,11,12,15,16],[4,8,9,13,14],[2,4,8,12,13,14,15,16],[1,3,5,6,7,9,10,11]};

%%
% mscohere [Cxy2, f] = mscohere(temp_after(ch1,:),temp_after(ch2,:),[],[],frequency{j},fs);
% ft_connectivity_wpli
% pn_eegPLV
for i = 1:size(range,2)
    for kind=1:2 %ASMR, SHAM
        for n=1:5
            load_before=load(['Sub' num2str(n) '_' num2str(kind) '1']); % load mat.file
            load_after=load(['Sub' num2str(n) '_' num2str(kind) '2']); % load mat.file

            %Channel select
            temp_before=load_before.Data(:,:,:);
            temp_after=load_after.Data(:,:,:);
            
            [wpli_before_temp] = WPLI2(temp_before,srate,range{i}(1),range{i}(2));
            [wpli_after_temp] = WPLI2(temp_after,srate,range{i}(1),range{i}(2));
            
            wpli_before_temp = reshape(mean(wpli_before_temp,1),19,19);
            wpli_after_temp = reshape(mean(wpli_after_temp,1),19,19);
                   
            wpli_before(:,:,n)=wpli_before_temp+wpli_before_temp';
            wpli_after(:,:,n)=wpli_after_temp+wpli_after_temp';
        end
    end
end
            