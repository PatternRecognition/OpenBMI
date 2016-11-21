function [ CF_PARAM ] = p300Training( FILE, fs )
%P300TRAINING Summary of this function goes here
%   Detailed explanation goes here

[eeg, eeg.mrk_orig, eeg.hdr]=Load_EEG_data(FILE,'device','brainVision','fs', fs);

mrk_define={'1','target','2','non_target'};

eeg.mrk=mrk_redefine_class(eeg.mrk_orig, mrk_define); 
eeg.x=eeg.x(:,1:6); %make functions -delete eog
eeg_epo=prep_segmentation(eeg, 'interval', [-200 800]);

eeg_epo=prep_baselineCorrection(eeg_epo, [0 400]);

eeg_epo.x=eeg_epo.x(40:200,:,:);

eeg_ft=func_featureExtraction(eeg_epo, 'ERPmean', 20);

[CF_PARAM]=classifier_trainClassifier(eeg_ft,'LDA');


end

