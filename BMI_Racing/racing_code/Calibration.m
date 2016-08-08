%% Accuracy, and parameters
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\DemoData'];
file=fullfile(BMI.EEG_DIR, '20160704_smkim_Run1NoFeedback');
band=[8 13];
fs=500;
[LOSS, CSP, LDA]=MI_calibration(file, band,fs);