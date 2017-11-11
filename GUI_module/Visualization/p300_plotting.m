function p300_plotting(filepath,subject_info, filename)

file=fullfile(filepath,subject_info.subject,subject_info.session,filename);

segTime=[-200 800];
baseTime=[-200 0];
selTime=[0 800];
nFeature=10;
Freq=[0.5 40];
if isequal(filename,'p300_on')
    marker={'1',1;'2',2;'3',3;'4',4;'5',5;'6',6;'7',7;'8',8;'9',9;'10',10;'11',11;'12',12};
    spellerText_on='PATTERN_RECOGNITION_MACHINE_LEARNING';
else
%     marker={'21','target';'22','nontarget'};      % s9 session1
    marker={'1','target';'2','nontarget'};
end
fs=100; 
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',fs});

cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
cnt=prep_selectChannels(cnt,{'Name',{'Cz', 'Oz'}});
cnt=prep_filter(cnt, {'frequency', Freq});

% transfer trigger to target/non-target
if isequal(filename,'p300_on')
    cnt=TriggerToTar_nTar(cnt,spellerText_on);
end

smt=prep_segmentation(cnt, {'interval', segTime});
smt=prep_baseline(smt, {'Time',baseTime});
smt=prep_selectTime(smt, {'Time',selTime});

% chanSel=[14 30];
% % for i=1:10
%     [LOSS, ~]=P300_calibration(file,filename,{'segTime',segTime;'baseTime',baseTime;'selTime',selTime;'nFeature',nFeature;'channel',chanSel;'Freq',Freq});
%     arr_loss(1)=LOSS{1,1};
% % end
% avg_loss=mean(arr_loss);
% Accuracy = 1- avg_loss;
Accuracy=NaN;
% accuracy with the number of correct characters
if isequal(filename,'p300_on')
    Accuracy = P300_Ncorrect(smt,filepath,subject_info,spellerText_on);
end
plot_x = 0:10:800;
t_c = smt.x(:,smt.y_logic(1,:),1);
t_o = smt.x(:,smt.y_logic(1,:),2);
n_c = smt.x(:,smt.y_logic(2,:),1);
n_o = smt.x(:,smt.y_logic(2,:),2);
t_c = mean(t_c,2);
t_o = mean(t_o,2);
n_c = mean(n_c,2);
n_o = mean(n_o,2);
f=figure;
y_l=[min([min(t_c), min(t_o),min(n_c), min(n_o)])*1.1 max([max(t_c), max(t_o),max(n_c), max(n_o)])*1.1];
subplot(2,1,1); plot(plot_x, t_c); hold on; plot(plot_x, n_c);
legend('target', 'non-target'); title('Cz'); ylim(y_l);
subplot(2,1,2); plot(plot_x, t_o); hold on; plot(plot_x, n_o);
legend('target', 'non-target'); title('Oz'); ylim(y_l);
a=suptitle(sprintf('%s / %s / %s / acc: %.2f%%',subject_info.subject,subject_info.session,filename,Accuracy*100));
set(a,'Interpreter','none');
saveas(f,sprintf('%s\\figure\\%s_%s_%s.jpg',filepath, subject_info.subject,subject_info.session,filename));
