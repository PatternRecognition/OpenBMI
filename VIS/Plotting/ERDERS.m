function ERDERS(filepath,subject_info, filename)
file=fullfile(filepath,subject_info.subject,subject_info.session, filename);

marker={'1','right';'2','left'};
fs=100; 
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',fs});
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);

%%
% laplacian
% channel selection
channel_index=[13, 14, 15]; % C3:52, Cz:54, C4:56
CNT=prep_selectChannels(CNT,{'Index',channel_index});
% band-pass filter
CNT=prep_filter(CNT, {'frequency', [8 13]}); % BPF 8~30 Hz
% envelop
% CNT=prep_envelope(CNT);
% segmentation
SMT=prep_segmentation(CNT, {'interval', [-490 4000]});
% envelop
SMT=prep_envelope(SMT);
% baseline
SMT=prep_baseline(SMT,{'Time',[-490 0]});
% class selection
SMT1=prep_selectClass(SMT,{'class',{'right'}});
SMT2=prep_selectClass(SMT,{'class',{'left'}});
% average
cls1_C3=mean(SMT1.x(:,:,1),2)';
cls2_C3=mean(SMT2.x(:,:,1),2)';
cls1_Cz=mean(SMT1.x(:,:,2),2)';
cls2_Cz=mean(SMT2.x(:,:,2),2)';
cls1_C4=mean(SMT1.x(:,:,3),2)';
cls2_C4=mean(SMT2.x(:,:,3),2)';

%% Making Classifier
band=[8 13]; fs=100; interval=[750 3500];
channel_index=1:44;% 귀채널 10개, EOG 2개, EMG 4개 제외하고,
arr_loss=zeros(10,1);
for i=1:10
    [LOSS, ~,~]=MI_calibration_2(file, band, fs, interval, {'nClass',2;'channel',channel_index});
    arr_loss(i)=LOSS{1,1};
end
avg_loss=mean(arr_loss);
Accuracy = 1- avg_loss;

%% plot
x=-490:10:4000;
y_l = [min([min(cls1_C3),min(cls1_Cz),min(cls1_C4),min(cls2_C3),min(cls2_Cz),min(cls2_C4)])*1.1...
    max([max(cls1_C3),max(cls1_Cz),max(cls1_C4),max(cls2_C3),max(cls2_Cz),max(cls2_C4)])*1.1];
f=figure;
subplot(1,3,1); plot(x,cls1_C3,'r',x,cls2_C3,'b');
title('C3'); legend('right', 'left'); xlim([-490 4000]); ylim(y_l);
subplot(1,3,2); plot(x,cls1_Cz,'r',x,cls2_Cz,'b');
title('Cz'); legend('right', 'left'); xlim([-490 4000]); ylim(y_l);
subplot(1,3,3); plot(x,cls1_C4,'r',x,cls2_C4,'b');
title('C4'); legend('right', 'left'); xlim([-490 4000]); ylim(y_l);
a=suptitle(sprintf('%s / %s / %s / acc: %.2f%%',subject_info.subject,subject_info.session,filename, Accuracy*100));
set(a,'Interpreter','none');
set(f, 'Position', [0, 500, 2000, 400]);
saveas(f,sprintf('%s\\figure\\%s_%s_%s.jpg',filepath, subject_info.subject,subject_info.session,filename));
