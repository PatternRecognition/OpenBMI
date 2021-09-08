N = 3;
P = excel_AUC(2,:);
% duration = (interval_sub(:,2) - interval_sub(:,1)) / 1000;
% duration = [1 1 1 1.5 1 1 1 1.5 1.5 1];
duration = 0.5;
C = 60./duration';

ITR = (log2(N)+P.*log2(P)+(1-P).*log2((1-P)./(N-1))).*C;
% disp(ITR)
mean(ITR)



