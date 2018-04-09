feature = [kss_s1 avg_PSD_s1];
feature = [kss_s1 feature2_avg_PSD_s1];
feature = [kss_s1 psd_feature_avg_s1];

feature = [kss_s1 avg_c_s1 avg_f_s1 avg_o_s1 avg_t_s1 avg_p_s1];
feature = [kss_s1 feature2_avg_PSD_c_s1 feature2_avg_PSD_f_s1 feature2_avg_PSD_o_s1 feature2_avg_PSD_t_s1 feature2_avg_PSD_p_s1];
feature = [kss_s1 psd_feautre_avg_c_s1 psd_feautre_avg_f_s1 psd_feautre_avg_o_s1 psd_feautre_avg_p_s1 psd_feautre_avg_t_s1];

feature = [kss_s1 avg_c1_s1 avg_c2_s1 avg_c3_s1 avg_f1_s1 avg_f2_s1 avg_f3_s1 avg_o1_s1 avg_o2_s1 avg_o3_s1 avg_t1_s1 avg_t2_s1 avg_p1_s1 avg_p2_s1 avg_p3_s1];
feature = [kss_s1 feature2_avg_PSD_c1_s1 feature2_avg_PSD_c2_s1 feature2_avg_PSD_c3_s1 feature2_avg_PSD_f1_s1 feature2_avg_PSD_f2_s1 feature2_avg_PSD_f3_s1 feature2_avg_PSD_o1_s1 feature2_avg_PSD_o2_s1 feature2_avg_PSD_o3_s1 feature2_avg_PSD_t1_s1 feature2_avg_PSD_t2_s1 feature2_avg_PSD_p1_s1 feature2_avg_PSD_p2_s1 feature2_avg_PSD_p3_s1];
feature = [kss_s1 psd_feautre_avg_c1_s1 psd_feautre_avg_c2_s1 psd_feautre_avg_c3_s1 psd_feautre_avg_f1_s1 psd_feautre_avg_f2_s1 psd_feautre_avg_f3_s1 psd_feautre_avg_o1_s1 psd_feautre_avg_o2_s1 psd_feautre_avg_o3_s1 psd_feautre_avg_p1_s1 psd_feautre_avg_p2_s1 psd_feautre_avg_p3_s1 psd_feautre_avg_t1_s1 psd_feautre_avg_t2_s1];


feature = [kss_s2 avg_PSD_s2];
feature = [kss_s2 feature2_avg_PSD_s2];
feature = [kss_s2 psd_feature_avg_s2];

feature = [kss_s2 avg_c_s2 avg_f_s2 avg_o_s2 avg_t_s2 avg_p_s2];
feature = [kss_s2 feature2_avg_PSD_c_s2 feature2_avg_PSD_f_s2 feature2_avg_PSD_o_s2 feature2_avg_PSD_t_s2 feature2_avg_PSD_p_s2];
feature = [kss_s2 psd_feautre_avg_c_s2 psd_feautre_avg_f_s2 psd_feautre_avg_o_s2 psd_feautre_avg_p_s2 psd_feautre_avg_t_s2];

feature = [kss_s2 avg_c1_s2 avg_c2_s2 avg_c3_s2 avg_f1_s2 avg_f2_s2 avg_f3_s2 avg_o1_s2 avg_o2_s2 avg_o3_s2 avg_t1_s2 avg_t2_s2 avg_p1_s2 avg_p2_s2 avg_p3_s2];
feature = [kss_s2 feature2_avg_PSD_c1_s2 feature2_avg_PSD_c2_s2 feature2_avg_PSD_c3_s2 feature2_avg_PSD_f1_s2 feature2_avg_PSD_f2_s2 feature2_avg_PSD_f3_s2 feature2_avg_PSD_o1_s2 feature2_avg_PSD_o2_s2 feature2_avg_PSD_o3_s2 feature2_avg_PSD_t1_s2 feature2_avg_PSD_t2_s2 feature2_avg_PSD_p1_s2 feature2_avg_PSD_p2_s2 feature2_avg_PSD_p3_s2];
feature = [kss_s2 psd_feautre_avg_c1_s2 psd_feautre_avg_c2_s2 psd_feautre_avg_c3_s2 psd_feautre_avg_f1_s2 psd_feautre_avg_f2_s2 psd_feautre_avg_f3_s2 psd_feautre_avg_o1_s2 psd_feautre_avg_o2_s2 psd_feautre_avg_o3_s2 psd_feautre_avg_p1_s2 psd_feautre_avg_p2_s2 psd_feautre_avg_p3_s2 psd_feautre_avg_t1_s2 psd_feautre_avg_t2_s2];


feature = [kss_s3 avg_PSD_s3];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s3 feature2_avg_PSD_s3];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s3 psd_feature_avg_s3];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s3 avg_c_s3 avg_f_s3 avg_o_s3 avg_t_s3 avg_p_s3];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4




feature = [kss_s3 feature2_avg_PSD_c_s3 feature2_avg_PSD_f_s3 feature2_avg_PSD_o_s3 feature2_avg_PSD_t_s3 feature2_avg_PSD_p_s3];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s3 psd_feautre_avg_c_s3 psd_feautre_avg_f_s3 psd_feautre_avg_o_s3 psd_feautre_avg_p_s3 psd_feautre_avg_t_s3];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s3 avg_c1_s3 avg_c2_s3 avg_c3_s3 avg_f1_s3 avg_f2_s3 avg_f3_s3 avg_o1_s3 avg_o2_s3 avg_o3_s3 avg_t1_s3 avg_t2_s3 avg_p1_s3 avg_p2_s3 avg_p3_s3];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s3 feature2_avg_PSD_c1_s3 feature2_avg_PSD_c2_s3 feature2_avg_PSD_c3_s3 feature2_avg_PSD_f1_s3 feature2_avg_PSD_f2_s3 feature2_avg_PSD_f3_s3 feature2_avg_PSD_o1_s3 feature2_avg_PSD_o2_s3 feature2_avg_PSD_o3_s3 feature2_avg_PSD_t1_s3 feature2_avg_PSD_t2_s3 feature2_avg_PSD_p1_s3 feature2_avg_PSD_p2_s3 feature2_avg_PSD_p3_s3];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s3 psd_feautre_avg_c1_s3 psd_feautre_avg_c2_s3 psd_feautre_avg_c3_s3 psd_feautre_avg_f1_s3 psd_feautre_avg_f2_s3 psd_feautre_avg_f3_s3 psd_feautre_avg_o1_s3 psd_feautre_avg_o2_s3 psd_feautre_avg_o3_s3 psd_feautre_avg_p1_s3 psd_feautre_avg_p2_s3 psd_feautre_avg_p3_s3 psd_feautre_avg_t1_s3 psd_feautre_avg_t2_s3];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4




feature = [kss_s4 avg_PSD_s4];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s4 feature2_avg_PSD_s4];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s4 psd_feature_avg_s4];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s4 avg_c_s4 avg_f_s4 avg_o_s4 avg_t_s4 avg_p_s4];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s4 feature2_avg_PSD_c_s4 feature2_avg_PSD_f_s4 feature2_avg_PSD_o_s4 feature2_avg_PSD_t_s4 feature2_avg_PSD_p_s4];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s4 psd_feautre_avg_c_s4 psd_feautre_avg_f_s4 psd_feautre_avg_o_s4 psd_feautre_avg_p_s4 psd_feautre_avg_t_s4];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s4 avg_c1_s4 avg_c2_s4 avg_c3_s4 avg_f1_s4 avg_f2_s4 avg_f3_s4 avg_o1_s4 avg_o2_s4 avg_o3_s4 avg_t1_s4 avg_t2_s4 avg_p1_s4 avg_p2_s4 avg_p3_s4];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s4 feature2_avg_PSD_c1_s4 feature2_avg_PSD_c2_s4 feature2_avg_PSD_c3_s4 feature2_avg_PSD_f1_s4 feature2_avg_PSD_f2_s4 feature2_avg_PSD_f3_s4 feature2_avg_PSD_o1_s4 feature2_avg_PSD_o2_s4 feature2_avg_PSD_o3_s4 feature2_avg_PSD_t1_s4 feature2_avg_PSD_t2_s4 feature2_avg_PSD_p1_s4 feature2_avg_PSD_p2_s4 feature2_avg_PSD_p3_s4];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s4 psd_feautre_avg_c1_s4 psd_feautre_avg_c2_s4 psd_feautre_avg_c3_s4 psd_feautre_avg_f1_s4 psd_feautre_avg_f2_s4 psd_feautre_avg_f3_s4 psd_feautre_avg_o1_s4 psd_feautre_avg_o2_s4 psd_feautre_avg_o3_s4 psd_feautre_avg_p1_s4 psd_feautre_avg_p2_s4 psd_feautre_avg_p3_s4 psd_feautre_avg_t1_s4 psd_feautre_avg_t2_s4];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4




feature = [kss_s5 avg_PSD_s5];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s5 feature2_avg_PSD_s5];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s5 psd_feature_avg_s5];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s5 avg_c_s5 avg_f_s5 avg_o_s5 avg_t_s5 avg_p_s5];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s5 feature2_avg_PSD_c_s5 feature2_avg_PSD_f_s5 feature2_avg_PSD_o_s5 feature2_avg_PSD_t_s5 feature2_avg_PSD_p_s5];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s5 psd_feautre_avg_c_s5 psd_feautre_avg_f_s5 psd_feautre_avg_o_s5 psd_feautre_avg_p_s5 psd_feautre_avg_t_s5];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s5 avg_c1_s5 avg_c2_s5 avg_c3_s5 avg_f1_s5 avg_f2_s5 avg_f3_s5 avg_o1_s5 avg_o2_s5 avg_o3_s5 avg_t1_s5 avg_t2_s5 avg_p1_s5 avg_p2_s5 avg_p3_s5];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s5 feature2_avg_PSD_c1_s5 feature2_avg_PSD_c2_s5 feature2_avg_PSD_c3_s5 feature2_avg_PSD_f1_s5 feature2_avg_PSD_f2_s5 feature2_avg_PSD_f3_s5 feature2_avg_PSD_o1_s5 feature2_avg_PSD_o2_s5 feature2_avg_PSD_o3_s5 feature2_avg_PSD_t1_s5 feature2_avg_PSD_t2_s5 feature2_avg_PSD_p1_s5 feature2_avg_PSD_p2_s5 feature2_avg_PSD_p3_s5];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s5 psd_feautre_avg_c1_s5 psd_feautre_avg_c2_s5 psd_feautre_avg_c3_s5 psd_feautre_avg_f1_s5 psd_feautre_avg_f2_s5 psd_feautre_avg_f3_s5 psd_feautre_avg_o1_s5 psd_feautre_avg_o2_s5 psd_feautre_avg_o3_s5 psd_feautre_avg_p1_s5 psd_feautre_avg_p2_s5 psd_feautre_avg_p3_s5 psd_feautre_avg_t1_s5 psd_feautre_avg_t2_s5];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4




feature = [kss_s6 avg_PSD_s6];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s6 feature2_avg_PSD_s6];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s6 psd_feature_avg_s6];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s6 avg_c_s6 avg_f_s6 avg_o_s6 avg_t_s6 avg_p_s6];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s6 feature2_avg_PSD_c_s6 feature2_avg_PSD_f_s6 feature2_avg_PSD_o_s6 feature2_avg_PSD_t_s6 feature2_avg_PSD_p_s6];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s6 psd_feautre_avg_c_s6 psd_feautre_avg_f_s6 psd_feautre_avg_o_s6 psd_feautre_avg_p_s6 psd_feautre_avg_t_s6];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4




feature = [kss_s6 avg_c1_s6 avg_c2_s6 avg_c3_s6 avg_f1_s6 avg_f2_s6 avg_f3_s6 avg_o1_s6 avg_o2_s6 avg_o3_s6 avg_t1_s6 avg_t2_s6 avg_p1_s6 avg_p2_s6 avg_p3_s6];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s6 feature2_avg_PSD_c1_s6 feature2_avg_PSD_c2_s6 feature2_avg_PSD_c3_s6 feature2_avg_PSD_f1_s6 feature2_avg_PSD_f2_s6 feature2_avg_PSD_f3_s6 feature2_avg_PSD_o1_s6 feature2_avg_PSD_o2_s6 feature2_avg_PSD_o3_s6 feature2_avg_PSD_t1_s6 feature2_avg_PSD_t2_s6 feature2_avg_PSD_p1_s6 feature2_avg_PSD_p2_s6 feature2_avg_PSD_p3_s6];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s6 psd_feautre_avg_c1_s6 psd_feautre_avg_c2_s6 psd_feautre_avg_c3_s6 psd_feautre_avg_f1_s6 psd_feautre_avg_f2_s6 psd_feautre_avg_f3_s6 psd_feautre_avg_o1_s6 psd_feautre_avg_o2_s6 psd_feautre_avg_o3_s6 psd_feautre_avg_p1_s6 psd_feautre_avg_p2_s6 psd_feautre_avg_p3_s6 psd_feautre_avg_t1_s6 psd_feautre_avg_t2_s6];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4





feature = [kss_s7 avg_PSD_s7];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s7 feature2_avg_PSD_s7];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s7 psd_feature_avg_s7];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s7 avg_c_s7 avg_f_s7 avg_o_s7 avg_t_s7 avg_p_s7];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s7 feature2_avg_PSD_c_s7 feature2_avg_PSD_f_s7 feature2_avg_PSD_o_s7 feature2_avg_PSD_t_s7 feature2_avg_PSD_p_s7];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s7 psd_feautre_avg_c_s7 psd_feautre_avg_f_s7 psd_feautre_avg_o_s7 psd_feautre_avg_p_s7 psd_feautre_avg_t_s7];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s7 avg_c1_s7 avg_c2_s7 avg_c3_s7 avg_f1_s7 avg_f2_s7 avg_f3_s7 avg_o1_s7 avg_o2_s7 avg_o3_s7 avg_t1_s7 avg_t2_s7 avg_p1_s7 avg_p2_s7 avg_p3_s7];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s7 feature2_avg_PSD_c1_s7 feature2_avg_PSD_c2_s7 feature2_avg_PSD_c3_s7 feature2_avg_PSD_f1_s7 feature2_avg_PSD_f2_s7 feature2_avg_PSD_f3_s7 feature2_avg_PSD_o1_s7 feature2_avg_PSD_o2_s7 feature2_avg_PSD_o3_s7 feature2_avg_PSD_t1_s7 feature2_avg_PSD_t2_s7 feature2_avg_PSD_p1_s7 feature2_avg_PSD_p2_s7 feature2_avg_PSD_p3_s7];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s7 psd_feautre_avg_c1_s7 psd_feautre_avg_c2_s7 psd_feautre_avg_c3_s7 psd_feautre_avg_f1_s7 psd_feautre_avg_f2_s7 psd_feautre_avg_f3_s7 psd_feautre_avg_o1_s7 psd_feautre_avg_o2_s7 psd_feautre_avg_o3_s7 psd_feautre_avg_p1_s7 psd_feautre_avg_p2_s7 psd_feautre_avg_p3_s7 psd_feautre_avg_t1_s7 psd_feautre_avg_t2_s7];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4





feature = [kss_s8 avg_PSD_s8];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s8 feature2_avg_PSD_s8];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s8 psd_feature_avg_s8];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s8 avg_c_s8 avg_f_s8 avg_o_s8 avg_t_s8 avg_p_s8];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s8 feature2_avg_PSD_c_s8 feature2_avg_PSD_f_s8 feature2_avg_PSD_o_s8 feature2_avg_PSD_t_s8 feature2_avg_PSD_p_s8];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s8 psd_feautre_avg_c_s8 psd_feautre_avg_f_s8 psd_feautre_avg_o_s8 psd_feautre_avg_p_s8 psd_feautre_avg_t_s8];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s8 avg_c1_s8 avg_c2_s8 avg_c3_s8 avg_f1_s8 avg_f2_s8 avg_f3_s8 avg_o1_s8 avg_o2_s8 avg_o3_s8 avg_t1_s8 avg_t2_s8 avg_p1_s8 avg_p2_s8 avg_p3_s8];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s8 feature2_avg_PSD_c1_s8 feature2_avg_PSD_c2_s8 feature2_avg_PSD_c3_s8 feature2_avg_PSD_f1_s8 feature2_avg_PSD_f2_s8 feature2_avg_PSD_f3_s8 feature2_avg_PSD_o1_s8 feature2_avg_PSD_o2_s8 feature2_avg_PSD_o3_s8 feature2_avg_PSD_t1_s8 feature2_avg_PSD_t2_s8 feature2_avg_PSD_p1_s8 feature2_avg_PSD_p2_s8 feature2_avg_PSD_p3_s8];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s8 psd_feautre_avg_c1_s8 psd_feautre_avg_c2_s8 psd_feautre_avg_c3_s8 psd_feautre_avg_f1_s8 psd_feautre_avg_f2_s8 psd_feautre_avg_f3_s8 psd_feautre_avg_o1_s8 psd_feautre_avg_o2_s8 psd_feautre_avg_o3_s8 psd_feautre_avg_p1_s8 psd_feautre_avg_p2_s8 psd_feautre_avg_p3_s8 psd_feautre_avg_t1_s8 psd_feautre_avg_t2_s8];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4




feature = [kss_s9 avg_PSD_s9];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s9 feature2_avg_PSD_s9];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s9 psd_feature_avg_s9];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s9 avg_c_s9 avg_f_s9 avg_o_s9 avg_t_s9 avg_p_s9];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s9 feature2_avg_PSD_c_s9 feature2_avg_PSD_f_s9 feature2_avg_PSD_o_s9 feature2_avg_PSD_t_s9 feature2_avg_PSD_p_s9];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s9 psd_feautre_avg_c_s9 psd_feautre_avg_f_s9 psd_feautre_avg_o_s9 psd_feautre_avg_p_s9 psd_feautre_avg_t_s9];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s9 avg_c1_s9 avg_c2_s9 avg_c3_s9 avg_f1_s9 avg_f2_s9 avg_f3_s9 avg_o1_s9 avg_o2_s9 avg_o3_s9 avg_t1_s9 avg_t2_s9 avg_p1_s9 avg_p2_s9 avg_p3_s9];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s9 feature2_avg_PSD_c1_s9 feature2_avg_PSD_c2_s9 feature2_avg_PSD_c3_s9 feature2_avg_PSD_f1_s9 feature2_avg_PSD_f2_s9 feature2_avg_PSD_f3_s9 feature2_avg_PSD_o1_s9 feature2_avg_PSD_o2_s9 feature2_avg_PSD_o3_s9 feature2_avg_PSD_t1_s9 feature2_avg_PSD_t2_s9 feature2_avg_PSD_p1_s9 feature2_avg_PSD_p2_s9 feature2_avg_PSD_p3_s9];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s9 psd_feautre_avg_c1_s9 psd_feautre_avg_c2_s9 psd_feautre_avg_c3_s9 psd_feautre_avg_f1_s9 psd_feautre_avg_f2_s9 psd_feautre_avg_f3_s9 psd_feautre_avg_o1_s9 psd_feautre_avg_o2_s9 psd_feautre_avg_o3_s9 psd_feautre_avg_p1_s9 psd_feautre_avg_p2_s9 psd_feautre_avg_p3_s9 psd_feautre_avg_t1_s9 psd_feautre_avg_t2_s9];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4




feature = [kss_s10 avg_PSD_s10];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s10 feature2_avg_PSD_s10];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s10 psd_feature_avg_s10];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s10 avg_c_s10 avg_f_s10 avg_o_s10 avg_t_s10 avg_p_s10];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s10 feature2_avg_PSD_c_s10 feature2_avg_PSD_f_s10 feature2_avg_PSD_o_s10 feature2_avg_PSD_t_s10 feature2_avg_PSD_p_s10];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s10 psd_feautre_avg_c_s10 psd_feautre_avg_f_s10 psd_feautre_avg_o_s10 psd_feautre_avg_p_s10 psd_feautre_avg_t_s10];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s10 avg_c1_s10 avg_c2_s10 avg_c3_s10 avg_f1_s10 avg_f2_s10 avg_f3_s10 avg_o1_s10 avg_o2_s10 avg_o3_s10 avg_t1_s10 avg_t2_s10 avg_p1_s10 avg_p2_s10 avg_p3_s10];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s10 feature2_avg_PSD_c1_s10 feature2_avg_PSD_c2_s10 feature2_avg_PSD_c3_s10 feature2_avg_PSD_f1_s10 feature2_avg_PSD_f2_s10 feature2_avg_PSD_f3_s10 feature2_avg_PSD_o1_s10 feature2_avg_PSD_o2_s10 feature2_avg_PSD_o3_s10 feature2_avg_PSD_t1_s10 feature2_avg_PSD_t2_s10 feature2_avg_PSD_p1_s10 feature2_avg_PSD_p2_s10 feature2_avg_PSD_p3_s10];



feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4



feature = [kss_s10 psd_feautre_avg_c1_s10 psd_feautre_avg_c2_s10 psd_feautre_avg_c3_s10 psd_feautre_avg_f1_s10 psd_feautre_avg_f2_s10 psd_feautre_avg_f3_s10 psd_feautre_avg_o1_s10 psd_feautre_avg_o2_s10 psd_feautre_avg_o3_s10 psd_feautre_avg_p1_s10 psd_feautre_avg_p2_s10 psd_feautre_avg_p3_s10 psd_feautre_avg_t1_s10 psd_feautre_avg_t2_s10];




feature = sortrows(feature,1);

for i=1:round(max(kss)-min(kss))
    kss_k_l(1) = 1;
    kss_k = find(min(feature(:,1))+i-1<=feature(:,1)&feature(:,1)<min(feature(:,1))+i)==1;
    kss_k_l(i+1) = length(kss_k);
    kss_k_l(i+1) = kss_k_l(i)+kss_k_l(i+1);
end

target = feature(kss_k_l(1):(kss_k_l(2)-1),2:end);
nonTarget = feature(kss_k_l(1):(kss_k_l(2)-1),1);
clear trainTarget trainNon testTarget testNon
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

train1_1 = target(trainTarget(1,:),:);
trainNon1_1 = nonTarget(trainNon(1,:),:);
testTarget1_1 = target(testTarget(1,:),:);
testNon1_1 = nonTarget(testNon(1,:),:);

train2_1 = target(trainTarget(2,:),:);
trainNon2_1 = nonTarget(trainNon(2,:),:);
testTarget2_1 = target(testTarget(2,:),:);
testNon2_1 = nonTarget(testNon(2,:),:);

train3_1 = target(trainTarget(3,:),:);
trainNon3_1 = nonTarget(trainNon(3,:),:);
testTarget3_1 = target(testTarget(3,:),:);
testNon3_1 = nonTarget(testNon(3,:),:);

train4_1 = target(trainTarget(4,:),:);
trainNon4_1 = nonTarget(trainNon(4,:),:);
testTarget4_1 = target(testTarget(4,:),:);
testNon4_1 = nonTarget(testNon(4,:),:);

for ii=2:round(max(kss)-min(kss))
    clear trainTarget trainNon testTarget testNon target nonTarget
    target = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),2:end);
    nonTarget = feature(kss_k_l(ii):(kss_k_l(ii+1)-1),1);
    
    trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
    testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];
    
    trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
    testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];
    
    trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
    testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];
    
    trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
    trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
    testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
    testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
    
    train1 = target(trainTarget(1,:),:);
    trainNon1 = nonTarget(trainNon(1,:),:);
    testTarget1 = target(testTarget(1,:),:);
    testNon1 = nonTarget(testNon(1,:),:);
    
    train2 = target(trainTarget(2,:),:);
    trainNon2 = nonTarget(trainNon(2,:),:);
    testTarget2 = target(testTarget(2,:),:);
    testNon2 = nonTarget(testNon(2,:),:);
    
    train3 = target(trainTarget(3,:),:);
    trainNon3 = nonTarget(trainNon(3,:),:);
    testTarget3 = target(testTarget(3,:),:);
    testNon3 = nonTarget(testNon(3,:),:);
    
    train4 = target(trainTarget(4,:),:);
    trainNon4 = nonTarget(trainNon(4,:),:);
    testTarget4 = target(testTarget(4,:),:);
    testNon4 = nonTarget(testNon(4,:),:);
    
    train1_1 = cat(1,train1_1,train1);
    trainNon1_1 = cat(1,trainNon1_1,trainNon1);
    testTarget1_1 = cat(1,testTarget1_1,testTarget1);
    testNon1_1 = cat(1,testNon1_1,testNon1);

    train2_1 = cat(1,train2_1,train2);
    trainNon2_1 = cat(1,trainNon2_1,trainNon2);
    testTarget2_1 = cat(1,testTarget2_1,testTarget2);
    testNon2_1 = cat(1,testNon2_1,testNon2);
    
    train3_1 = cat(1,train3_1,train3);
    trainNon3_1 = cat(1,trainNon3_1,trainNon3);
    testTarget3_1 = cat(1,testTarget3_1,testTarget3);
    testNon3_1 = cat(1,testNon3_1,testNon3);

    train4_1 = cat(1,train4_1,train4);
    trainNon4_1 = cat(1,trainNon4_1,trainNon4);
    testTarget4_1 = cat(1,testTarget4_1,testTarget4);
    testNon4_1 = cat(1,testNon4_1,testNon4);

end
xtrain_1 = [train1_1];
Xtrain_1 = [ones(length(xtrain_1),1),xtrain_1];
ytrain_1 = [trainNon1_1];
beta_1 = mvregress(Xtrain_1,ytrain_1);

xtest_1 = [testTarget1_1];
Xtest_1 = [ones(length(xtest_1),1),xtest_1];
ytest_1 = [testNon1_1];
y_hat_1 = Xtest_1*beta_1;

for i=1:length(y_hat_1)
    prediction_matrix= y_hat_1-1.5<ytest_1 & ytest_1<y_hat_1+1.5;
end

prediction_1 = find(prediction_matrix==1);
prediction_eeg_sleep_1 = length(prediction_1)/length(y_hat_1)*100;
RMSE_1 = mean(sqrt((ytest_1-y_hat_1).^2/(length(ytest_1))));
Error_Rate_1 = sum(abs(ytest_1-y_hat_1)) / length(ytest_1);

xtrain_2 = [train2_1];
Xtrain_2 = [ones(length(xtrain_2),1),xtrain_2];
ytrain_2 = [trainNon2_1];
beta_2 = mvregress(Xtrain_2,ytrain_2);

xtest_2 = [testTarget2_1];
Xtest_2 = [ones(length(xtest_2),1),xtest_2];
ytest_2 = [testNon2_1];
y_hat_2 = Xtest_2*beta_2;


for i=1:length(y_hat_2)
    prediction_matrix= y_hat_2-1.5<ytest_2 & ytest_2<y_hat_2+1.5;
end

prediction_2 = find(prediction_matrix==1);
prediction_eeg_sleep_2 = length(prediction_2)/length(y_hat_2)*100;
RMSE_2 = mean(sqrt((ytest_2-y_hat_2).^2/(length(ytest_2))));
Error_Rate_2 = sum(abs(ytest_2-y_hat_2)) / length(ytest_2);

xtrain_3 = [train3_1];
Xtrain_3 = [ones(length(xtrain_3),1),xtrain_3];
ytrain_3 = [trainNon3_1];
beta_3 = mvregress(Xtrain_3,ytrain_3);

xtest_3 = [testTarget3_1];
Xtest_3 = [ones(length(xtest_3),1),xtest_3];
ytest_3 = [testNon3_1];
y_hat_3 = Xtest_3*beta_3;


for i=1:length(y_hat_3)
    prediction_matrix= y_hat_3-1.5<ytest_3 & ytest_3<y_hat_3+1.5;
end

prediction_3 = find(prediction_matrix==1);
prediction_eeg_sleep_3 = length(prediction_3)/length(y_hat_3)*100;
RMSE_3 = mean(sqrt((ytest_3-y_hat_3).^2/(length(ytest_3))));
Error_Rate_3 = sum(abs(ytest_3-y_hat_3)) / length(ytest_3);


xtrain_4 = [train4_1];
Xtrain_4 = [ones(length(xtrain_4),1),xtrain_4];
ytrain_4 = [trainNon4_1];
beta_4 = mvregress(Xtrain_4,ytrain_4);

xtest_4 = [testTarget4_1];
Xtest_4 = [ones(length(xtest_4),1),xtest_4];
ytest_4 = [testNon4_1];
y_hat_4 = Xtest_4*beta_4;


for i=1:length(y_hat_4)
    prediction_matrix= y_hat_4-1.5<ytest_4 & ytest_4<y_hat_4+1.5;
end

prediction_4 = find(prediction_matrix==1);
prediction_eeg_sleep_4 = length(prediction_4)/length(y_hat_4)*100;
RMSE_4 = mean(sqrt((ytest_4-y_hat_4).^2/(length(ytest_4))));
Error_Rate_4 = sum(abs(ytest_4-y_hat_4)) / length(ytest_4);



prediction_eeg_sleep = (prediction_eeg_sleep_1 + prediction_eeg_sleep_2 + prediction_eeg_sleep_3 + prediction_eeg_sleep_4) / 4
RMSE = (RMSE_1 + RMSE_2 + RMSE_3 + RMSE_4) / 4
Error_Rate = (Error_Rate_1 + Error_Rate_2 + Error_Rate_3 + Error_Rate_4) / 4

