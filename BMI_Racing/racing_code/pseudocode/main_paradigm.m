
%% Basic Setting
band=[7 13];fs=1000;

% fold='C:\Users\CVPR\Desktop\EEG\OpenBMI\data'
% %% practicing with visual ERD/S
% visual_ERD_on(band,[0 20] ,2); % 1: envelop, 2: power
% 
% %% Offline raining
% Makeparadigm_MI_fb({'time_sti',5; 'time_isi',3; 'num_trial',50; 'classes',{'right','left', 'foot','rest'}; 'time_jitter',0.1; 'screen','window'; 'TCPIP','on'});
% visual_Paradigm_on([], [500 3500], [11 20] ); %ch, ival, band, varargin

%% Classifier Training & Visualization
file='C:\Users\CVPR\Desktop\EEG\OpenBMI\data\2016_07_12_mhlee_training';
[CSP_W, CF_PARAM, loss] = racing_calibration(file, band, fs, {'visualization' , 1}); % 1: visual on, 0: off


%% Pseudo-online emulator
file='C:\Users\CVPR\Desktop\EEG\OpenBMI\data\2016_07_12_mhlee_test1';
marker= {'1','right';'2','left';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', fs});
cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, {'x','t','fs','y_dec','y_logic','y_class','class', 'chan'});

racing_pseudoOnline(cnt,{'band',band;'fs',fs;...
    'bufferSize',5;'windowSize',3;'stepSize',0.5;...
    'classifier',CF_PARAM;'feature',CSP_W;'clf',7:10});

[fb_loss,ox,trueLabel,clfLabel]=racing_pseudoOnline3(cnt,{'band',band;'fs',fs;...
    'bufferSize',5;'windowSize',3;'stepSize',0.5;'topo',0;...
    'classifier',CF_PARAM;'feature',CSP_W;'clf',7:10}); % OVR ºÐ·ù±â (7:10)


%% Tuning
visual_CFY_on(CSP_W, CF_PARAM, band)

%% Racing Game
racing_on2(CSP_W, CF_PARAM, band);
visual_CFY_on(CSP_W, CF_PARAM, band)

%%












%% training
for j=1:3
    for i=1:4
        pause(2)
        [cf_dat{j, i}]=save_CFY(CSP_W, CF_PARAM, band, 7);
        pause(5)
    end
end


for i=1:4
    tm{i}=cf_dat{i}(length(cf_dat{i})-300+1:end,:)
end
x=[tm{1}' tm{2}' tm{3}' tm{4}']
y=[];tm2=[];
for i=1:4
tm2(1:length(tm{1}))=i;
y=[y tm2]
end
cly.x=x;
cly.y_logic=logical(1200)
cly.y_logic(1,1:300)=1;cly.y_logic(2,301:600)=1;cly.y_logic(3,601:900)=1;cly.y_logic(4,901:1200)=1;
cly.y_dec=y
cly.class={'1', 'right';'2', 'left'; '3', 'foot'; '4', 'rest'};
cly.y_class=[];
for i=1:4
    fv1=changeLabels(cly,{cly.class{i,2},1;'others',2});  
    [CF_LDA{i}]=func_train(fv1,{'classifier','LDA'});
end
%%

%% test
for i=1:4
    pause(2)
    [cf_test{i}]=save_CFY(CSP_W, CF_PARAM, band, 10);
    pause(5)
end
for i=1:4
    tm3{i}=cf_test{i}(length(cf_test{i})-300+1:end,:)
end
x2=[tm3{1}' tm3{2}' tm3{3}' tm3{4}']


for i=1:length(x2(1,:))
    for j=1:4
        [cf_out(i,j)]=func_predict(x2(:,i), CF_LDA{j});
    end
end
[out b]=min(cf_out');


[out b]=min(cf_out')

visual_CFY_on(CSP_W, CF_PARAM, band, CF_LDA)
th(1)=-2, th(2)=-4, th(3)=-2, th(4)=-5;
racing_on(CSP_W, CF_PARAM, band, th, CF_LDA);


racing_on2(CSP_W, CF_PARAM, band);
visual_CFY_on(CSP_W, CF_PARAM, band)







% for b=3.5:-0.5:3
%     for w=4:-0.5:1
%         if b>w
%             loss=racing_pseudoOnline3(fold,cnt,{'band',band;'fs',fs;...
%                 'interval',[750 3500];'bufferSize',b;'windowSize',w;'stepSize',0.5;...
%                 'classifier',CF_PARAM;'feature',CSP_W;'clf',[7:10];'topo',0});
%             str=sprintf('bufferSize: %.1f(s),  windowSize: %.1f(s),  loss: %.3f', b,w,loss);
%             disp(str)
%         end
%     end
% end





