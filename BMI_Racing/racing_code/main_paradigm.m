clc
close all;
clear all;

%% Basic Setting
band=[13 21];fs=1000;
fold='G:\data2'

%% 1. practicing with visual ERD/S
visual_ERD_on(band,[0 20] ,2, {'C3', 'Cz', 'C4'}); % 1: envelop, 2: power

%% 2. Offline raining- matlab1 and matlab 2
% we can use the paradigm solely if the parameter ['TCPIP', 'off'], while
% in the case of ['TCPIP', 'on'], you should execute the
% 'visual_Paradigm_on' at matlab two
Makeparadigm_MI_fb({'time_sti',4; 'time_isi',4; 'num_trial',50; 'classes',{'right','left', 'foot','rest'}; 'time_jitter',0.1; 'screen','full'; 'TCPIP','off'});
[trial, CSP, CLY_LDA]=visual_Paradigm_on([], [500 3500], [11 20] ); %ch, ival, band, varargin , % csp, and lda parrameters from the online data

%% 3. Classifier Training & Visualization
file='G:\data2\2016_07_28_hkkim_short_training'
[CSP_W, CF_PARAM, loss] = racing_calibration(file, band, fs, {'visualization' , 1}); % 1: visual on, 0: off

%% 4. Pseudo-online data acquisition
Makeparadigm_MI_fb({'time_sti',8; 'time_isi',3; 'num_trial',5; 'classes',{'right','left', 'foot','rest'}; 'time_jitter',0.1; 'screen','full'; 'TCPIP','off'});

%% 5. Pseudo-online emulator - check the STI and short training data
file='G:\data2\2016_07_28_hkkim_short_training'
marker= {'1','right';'2','left';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', fs});
cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, {'x','t','fs','y_dec','y_logic','y_class','class', 'chan'});

%% 6. ovr �з� ��� �� plot�� �׸��� true label ǥ��
racing_pseudoOnline(cnt,{'band',band;'fs',fs;...
    'bufferSize',5;'windowSize',3;'stepSize',0.5;...
    'classifier',CF_PARAM;'feature',CSP_W;'clf',7:10});

[fb_loss,ox,trueLabel,clfLabel, cf_out, t]=racing_pseudoOnline3(cnt,{'band',band;'fs',fs;...
    'bufferSize',5;'windowSize',3;'stepSize',0.5;'topo',0;...
    'classifier',CF_PARAM;'feature',CSP_W;'clf',1:10}); % OVR �з��� (7:10)

racing_plot(clfLabel,trueLabel,{'time',t});

%% 7. real-time paradigm
visual_CFY_on2(CSP_W, CF_PARAM, band);

visual_CFY_on(CSP_W, CF_PARAM, band)

%% 8. Racing Game
racing_on2(CSP_W, CF_PARAM, band);
visual_CFY_on(CSP_W, CF_PARAM, band)

%%
file='G:\data2\2016_07_28_hkkim_short_training'
[CF_PARAM2] = racing_calibration_temp(file, band, fs, {'visualization' , 1}); % 1: visual on, 0: off


%% Pseudo-online emulator
file='G:\data2\2016_07_28_hkkim_short_training'
marker= {'1','right';'2','left';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', fs});
cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, {'x','t','fs','y_dec','y_logic','y_class','class', 'chan'});
CSP_W=[]
% ovr �з� ��� �� plot�� �׸��� true label ǥ��
racing_pseudoOnline_temp(cnt,{'band',band;'fs',fs;...
    'bufferSize',5;'windowSize',3;'stepSize',0.5;...
    'classifier',CF_PARAM2;'feature',CSP_W;'clf',7:10});

[fb_loss,ox,trueLabel,clfLabel]=racing_pseudoOnline3_temp(cnt,{'band',band;'fs',fs;...
    'bufferSize',5;'windowSize',3;'stepSize',0.5;'topo',0;...
    'classifier',CF_PARAM2;'feature',CSP_W;'clf',7:10}); % OVR �з��� (7:10)



















figure
ch=[6 16 26]
for i=1:3
subplot(3,1,i)
[a b]=find(mCNT.y_dec==1)
stem(x(ch(i),b),'MarkerFaceColor',[1 0 0]);hold on;
mn(1, i)=mean(x(ch(i),b));
sd(1, i)=std(x(ch(i),b));

[a b]=find(mCNT.y_dec==2)
stem(x(ch(i),b),'MarkerFaceColor',[0 1 0]);hold on;
mn(2, i)=mean(x(ch(i),b));
sd(2, i)=std(x(ch(i),b));

[a b]=find(mCNT.y_dec==3)
stem(x(ch(i),b),'MarkerFaceColor',[0 0 1]);hold on;
mn(3, i)=mean(x(ch(i),b));
sd(3, i)=std(x(ch(i),b));

[a b]=find(mCNT.y_dec==4)
stem(x(ch(i),b),'MarkerFaceColor',[0.5 0.5 0.5]);hold on;
mn(4, i)=mean(x(ch(i),b));
sd(4, i)=std(x(ch(i),b));
end

for j=1:length(SMT.class)
for i=1:length(SMT.chan)
    [a b]=find(mCNT.y_dec==j)
    mn(j, i)=mean(x(i,b));
    sd(j, i)=std(x(i,b));    
end
end

for j=1:length(SMT.class)
for i=1:length(SMT.chan)
a_feature{j,i}=normrnd(mn(j,i),sd(j, i)/3,[1 100]);
end
end
a_feat=[]
for i=1:length(SMT.chan)
a_feat=cat(1,a_feat,a_feature{1,i})
end






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







for b=3.5:-0.5:3
    for w=4:-0.5:1
        if b>w
            loss=racing_pseudoOnline3(fold,cnt,{'band',band;'fs',fs;...
                'interval',[750 3500];'bufferSize',b;'windowSize',w;'stepSize',0.5;...
                'classifier',CF_PARAM;'feature',CSP_W;'clf',[7:10];'topo',0});
            str=sprintf('bufferSize: %.1f(s),  windowSize: %.1f(s),  loss: %.3f', b,w,loss);
            disp(str)
        end
    end
end


%% manually classier construction

tm=[]
x=[tm{1}' tm{2}' tm{3}' tm{4}']
y=[];tm2=[];
for i=1:4
tm2(1:length(tm{1}))=i;
y=[y tm2]
end
cly.x=tm';
cly.y_logic=logical(80)
cly.y_logic(1,1:20)=1;cly.y_logic(2,21:40)=1;cly.y_logic(3,41:60)=1;cly.y_logic(4,61:80)=1;
cly.class={'1', 'right';'2', 'left'; '3', 'foot'; '4', 'rest'};
cly.y_dec=[];
cly.y_class=[];
for i=1:4
    fv1=changeLabels(cly,{cly.class{i,2},1;'others',2});  
    [CF_LDA{i}]=func_train(fv1,{'classifier','LDA'});
end

[fb_loss,ox,trueLabel,clfLabel, cf_out]=racing_pseudoOnline3(cnt,{'band',band;'fs',fs;...
    'bufferSize',5;'windowSize',3;'stepSize',0.5;'topo',0;...
    'classifier',CF_PARAM;'feature',CSP_W;'clf',1:10; 'out_classifier',CF_LDA}); % OVR �з��� (7:10)