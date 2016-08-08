%% Training
clear all;close all; clc;
band=[7 13];fs=250;
% visual_ERD_on(band)
% Makeparadigm_MI({'time_sti',4,'time_isi',3,'time_rest',1,'num_trial',50,'num_class',4,'time_jitter',0.1,'num_screen',2});

%% calibration
file='C:\Users\Administrator\Desktop\EEG\20160722_hsan_1';
 [CSP_W, CF_PARAM, loss] = racing_calibration_(file, band, fs);

visual_CFY_on(CSP_W, CF_PARAM, band)

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











