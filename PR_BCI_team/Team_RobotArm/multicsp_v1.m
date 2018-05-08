clc; clear all; close all;


dd='E:\Code_overall\Data\Converted\Reaching\';
filelist={'20180208_jgyoon_reaching_MI'};

% 6 without Rest: by basic code : 32.5%

channel_layout=[8 9 10 11 13 14 15 18 19 20 21 43 44 47 48 49 50 52 53 54];

Result=zeros(length(filelist),1);
Result_std=zeros(length(filelist),1);
%%
for i=1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    ival=[0 3000];
    
    %% band pass filtering, order of 5, range of 8-15Hz
    cnt=proc_filtButter(cnt,2, [8 15]);
    cnt=proc_selectChannels(cnt, [8 9 10 11 13 14 15 18 19 20 21 43 44 47 48 49 50 52 53 54]);
   
    %% cnt to epoch
    epo=cntToEpo(cnt,mrk,ival);

    %% Extract the Rest class
    epoNRest=proc_selectClasses(epo,{'Forward','Backward','Left','Right','Up','Down'});
    epoRest=proc_selectClasses(epo,{'Rest'});
    
    %% extract the same amount of Rest as other classes
    epoRest.x=datasample(epoRest.x,40,3,'Replace',false);
    epoRest.y=datasample(epoRest.y,40,2,'Replace',false);
    
    %% concatenate the classes
    epo_all=proc_appendEpochs(epoNRest,epoRest);
    
    %% usual csp usage
%     %% CSP - FEATURE EXTRACTION
%     [csp_fv,csp_w,csp_eig]=proc_multicsp(epoNRest,3);
%     proc=struct('memo','csp_w');
%     
%     proc.train= ['[fv,csp_w]=  proc_multicsp(fv, 3); ' ...
%         'fv= proc_variance(fv); ' ...
%         'fv= proc_logarithm(fv);'];
%     
%     proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ','fv= proc_variance(fv); ' ,'fv= proc_logarithm(fv);'];
%     %% CLASSIFIER
%     [C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(epoNRest,'RLDAshrink','proc',proc, 'kfold', 5);
%     Result(i)=1-C_eeg;
%     Result_Std(i)=loss_eeg_std;
%     All_csp_w(:,:,i)=csp_w;
    %% multi csp without toolbox
    %input is epoNRest and the order
    dat=permute(epoNRest.x,[2 1 3]);
    [nChan,nTimes,nTrials]=size(dat);
    
    nFeat=size(dat,2)*size(dat,3);
    labels=epoNRest.y;
    nClasses=size(labels,1);
    %% way can be pairwise or one vs all or complete
    way='one-vs-all';
    %way='pairwise';
    
    switch way
        case 'one-vs-all'
            way=2*eye(nClasses)-ones(nClasses);
        case 'pairwise'
            way=[]
            for i =1:nClasses
                for j=i+1:nClasses;
                    vec=zeros(1,nClasses);
                    vec(i)=1;
                    vec(j)=-1;
                    way=[way;vec];
                end
            end
        case 'complete'
            a=dec2bin(1:2^(nClasses-1)-1,nClasses);
            way=[]
            for i=1:size(a,1)
                d=transpose(str2num(a(i,:)'));
                c=sum(d);
                if c>=2
                    b=dec2bin(1:2^(c-1)-1,c);
                    for j=1:size(b,1)
                        bb=transpose(str2num(b(j,:)'));
                        e=d;
                        e(find(d>0))=2*bb-1;
                        way=[way;e];
                    end
                end
            end
    end
    
    %% covariance matrix
    Sig=zeros(nChan,nChan,nClasses);
    for i = 1:nClasses
        da=dat(:,:,find(labels(i,:)>0)); %find the value of label 'i'
        % the data format is channel X time X trials
        da=da(:,:);
        % becomes channel X (time*trials)
        Sig(:,:,i)=da*da'/size(da,2);   %size(da,2)is the total length (time*trials)
    end
    nComps=1;
    w=zeros(nChan,size(way,1),nComps*2);
    la=zeros(size(way,1),nComps*2);
    
    %%
    for i=1:size(way,1) %size(way,1)== nClasses
        ind1=find(way(i,:)==1); %index of diagonal component of way
        ind2=find(way(i,:)==-1);%index of other component
        
        %covariance matrix of selected class
        Sig1=mean(Sig(:,:,ind1),3);
        %mean covariance matrix of other classes
        Sig2=mean(Sig(:,:,ind2),3);
        %diagonal matrix D of eigenvalues
        %full matrix P whose columns are the corresponding eigenvectors
        [P,D]=eig(Sig1+Sig2);
        
    end
end