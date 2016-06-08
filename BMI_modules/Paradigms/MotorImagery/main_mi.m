


%% Accuracy, and parameters
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\DemoData'];
file=fullfile(BMI.EEG_DIR, '\feedback_motorimageryVPkg');
[LOSS, CSP, LDA]=cal_loss(file);

%% calibration for feedback experiment
[weight, bias]=MI_calibration();
