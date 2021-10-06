% calculate AUC

for j=1:15
[~,~,~,AUC(j)] = perfcurve(ans_auc(:,j),prop(:,j),0);
end

mean_AUC = mean(AUC);
disp(mean_AUC)