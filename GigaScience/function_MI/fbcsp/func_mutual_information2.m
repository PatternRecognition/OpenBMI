function dat_out = func_mutual_information2(dat_in,dat_in1,dat_in2) 

% Check validity
% Dimension of input % only for 'vector'
% Length of input % same length ?

% pdf estimation


% calculation of mutual information

% Use external library
%%%%%%%%%%%%%%%%%%%%%%%%%%% Mutual info code 1
% echo on;
% 
% for ii=1:size(dat_in1,1)
%     %     mut_info(ii)=mutualinfo(dat_in1(ii,:),dat_in2);
%     %     mut_info2(ii)=MI(dat_in1(ii,:)',dat_in2');
%     mut_info3(ii)=mi(dat_in1(ii,:)',dat_in2');
% end
% 
% dat_out=mut_info3';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Mutual info code 2
%%
dat_in.y_dec;
label1=str2num(dat_in.class{1,1});
[a b] =find(dat_in.y_dec(:) == label1);
dat_in.y_dec(a) =1;

label2=str2num(dat_in.class{2,1});
[aa bb ] = find(dat_in.y_dec(:) == label2);
dat_in.y_dec(aa) =2;

%%
kernelWidth=1;
features=dat_in1;
labels=dat_in.y_dec;
[,idx_c1]=find(labels==1); [,idx_c2]=find(labels==2);
feat_c1=features(:,idx_c1); feat_c2=features(:,idx_c2);

for i=1:size(features,1)
    miValue(i) = proc_mutual_information( feat_c1(i,:), feat_c2(i,:), kernelWidth );
end

dat_out=miValue';
end

