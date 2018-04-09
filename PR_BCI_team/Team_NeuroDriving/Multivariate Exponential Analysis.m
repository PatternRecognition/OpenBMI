feature = [kss_s1 avg_PSD_s1];

feature = [kss_s1 avg_c_s1 avg_f_s1 avg_o_s1 avg_t_s1 avg_p_s1];
feature = [kss_s1 feature2_avg_PSD_c_s1 feature2_avg_PSD_f_s1 feature2_avg_PSD_o_s1 feature2_avg_PSD_t_s1 feature2_avg_PSD_p_s1];
feature = [kss_s1 psd_feautre_avg_c_s1 psd_feautre_avg_f_s1 psd_feautre_avg_o_s1 psd_feautre_avg_p_s1 psd_feautre_avg_t_s1];

feature = [kss_s1 avg_c1_s1 avg_c2_s1 avg_c3_s1 avg_f1_s1 avg_f2_s1 avg_f3_s1 avg_o1_s1 avg_o2_s1 avg_o3_s1 avg_t1_s1 avg_t2_s1 avg_p1_s1 avg_p2_s1 avg_p3_s1];
feature = [kss_s1 feature2_avg_PSD_c1_s1 feature2_avg_PSD_c2_s1 feature2_avg_PSD_c3_s1 feature2_avg_PSD_f1_s1 feature2_avg_PSD_f2_s1 feature2_avg_PSD_f3_s1 feature2_avg_PSD_o1_s1 feature2_avg_PSD_o2_s1 feature2_avg_PSD_o3_s1 feature2_avg_PSD_t1_s1 feature2_avg_PSD_t2_s1 feature2_avg_PSD_p1_s1 feature2_avg_PSD_p2_s1 feature2_avg_PSD_p3_s1];
feature = [kss_s1 psd_feautre_avg_c1_s1 psd_feautre_avg_c2_s1 psd_feautre_avg_c3_s1 psd_feautre_avg_f1_s1 psd_feautre_avg_f2_s1 psd_feautre_avg_f3_s1 psd_feautre_avg_o1_s1 psd_feautre_avg_o2_s1 psd_feautre_avg_o3_s1 psd_feautre_avg_p1_s1 psd_feautre_avg_p2_s1 psd_feautre_avg_p3_s1 psd_feautre_avg_t1_s1 psd_feautre_avg_t2_s1];

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

scatter(kss_s1, avg_PSD_s1(:,1),'g')
hold on
scatter(kss_s1, avg_PSD_s1(:,2),'r')
hold on
scatter(kss_s1, avg_PSD_s1(:,3),'b')
hold on
scatter(kss_s1, avg_PSD_s1(:,4),'k')
hold on
scatter(kss_s1, avg_PSD_s1(:,1),'y')
