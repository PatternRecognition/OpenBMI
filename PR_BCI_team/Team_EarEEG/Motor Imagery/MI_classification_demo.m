%%
clc
clear
OpenBMI('G:\Samsung\Toolbox')
dname=('D:\2017_0702_KYJ\GG\Dataset\competitionIII_IVa_MI');
% dname=('E:\starlapdata\experiment_data\subject18\Session1');

for tt=1:1
    for Subj=1:5
%         file=(sprintf('161205_skjeon_s1_realMove'));
%         file=fullfile(dname, file);
%         marker={'1','left';'2','right';'3','foot'};
%         [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});
%         field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
%         CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
%         
        file=(sprintf('%d.mat',Subj));
        
        fname=fullfile(dname,file);
        data=load(fname);
        CNT=compMI2ours(double(data.cnt), data.mrk, data.nfo);
        %%
        % k set
        k=5; % k-fold cross-validation
        m=length(CNT.t)/length(CNT.class); % trial / class = 50 이어야함
        for i=1:k
            kk{i}=(i-1)*m/k+1:m/k*i;
        end
        
        % class selection
        Cls1=prep_selectClass(CNT,{'class','right'});
        Cls2=prep_selectClass(CNT,{'class','foot'});
        
        % Random order array
%         mm=randperm(m);
         mm=[80,70,87,20,90,110,32,67,134,77,15,48,132,38,89,58,16,45,104,107,122,24,42,130,44,9,30,33,123,52,41,29,126,94,111,63,27,3,128,139,46,127,65,76,56,85,47,112,83,115,88,118,8,10,60,137,86,129,18,75,57,84,23,7,11,2,100,97,28,19,50,91,106,35,17,21,36,64,138,92,14,124,113,22,119,6,43,82,54,102,61,81,49,73,116,96,13,95,62,93,55,131,25,140,101,4,125,69,99,135,26,66,31,109,136,12,72,105,79,121,39,59,74,71,1,133,40,98,120,53,114,37,5,108,117,34,103,51,68,78];
%       
        temp1=Cls1; temp2=Cls2;
        for i=1:m
            Cls1.t(:,mm(i))=temp1.t(:,i);
            Cls2.t(:,mm(i))=temp2.t(:,i);
        end
        
        % Initial value
        Cls1_tr=Cls1; Cls1_te=Cls1;
        Cls2_tr=Cls2; Cls2_te=Cls2;
        
        % te:tr=1:k-1 data set
        zz=1:k;
        
        
        % CV 시작 빡대가리 ㅅㅂ ㅅ ㅂ ㅅ ㅂ ㅅ ㅂ ㄱ ㅐ빡대가리
        for qq=1:k
            temp=zz;
            for i=1:k
                if i<k
                    zz(i)=temp(i+1);
                else
                    zz(i)=temp(1);
                end
            end
            
%             zz=[1,2,3,4,5];
%             zz=[1,1,1,1,1];
%             zz=[3,4,5,1,2];
            
            te_set=cell2mat(kk(zz(1)));
            tr_set=cell2mat(kk(zz(2:k)));
            
            % test set
            Cls1_te.t=Cls1.t(te_set);
            Cls1_te.y_dec=Cls1.y_dec(te_set);
            Cls1_te.y_logic=Cls1.y_logic(te_set);
            Cls1_te.y_class=Cls1.y_class(te_set);
            
            Cls2_te.t=Cls2.t(te_set);
            Cls2_te.y_dec=Cls2.y_dec(te_set);
            Cls2_te.y_logic=Cls2.y_logic(te_set);
            Cls2_te.y_class=Cls2.y_class(te_set);
            
            % training set
            Cls1_tr.t=Cls1.t(tr_set);
            Cls1_tr.y_dec=Cls1.y_dec(tr_set);
            Cls1_tr.y_logic=Cls1.y_logic(tr_set);
            Cls1_tr.y_class=Cls1.y_class(tr_set);
            
            Cls2_tr.t=Cls2.t(tr_set);
            Cls2_tr.y_dec=Cls2.y_dec(tr_set);
            Cls2_tr.y_logic=Cls2.y_logic(tr_set);
            Cls2_tr.y_class=Cls2.y_class(tr_set);
            
            % CNT_tr=Cls1_tr Cls2_tr;
            CNT_tr=CNT;
            CNT_tr.t=[Cls1_tr.t Cls2_tr.t];
            CNT_tr.y_dec=[Cls1_tr.y_dec Cls2_tr.y_dec];
            CNT_tr.y_class=[Cls1_tr.y_class Cls2_tr.y_class];
            CNT_tr.y_logic=[Cls1_tr.y_logic zeros(1,length(Cls1_tr.y_logic));zeros(1,length(Cls1_tr.y_logic)) Cls2_tr.y_logic];
            CNT_tr.class=[Cls1_tr.class;Cls2_tr.class];
            % CNT_te=Cls1_te Cls2_te;
            CNT_te=CNT;
            CNT_te.t=[Cls1_te.t Cls2_te.t];
            CNT_te.y_dec=[Cls1_te.y_dec Cls2_te.y_dec];
            CNT_te.y_class=[Cls1_te.y_class Cls2_te.y_class];
            CNT_te.y_logic=[Cls1_te.y_logic zeros(1,length(Cls1_te.y_logic));zeros(1,length(Cls1_te.y_logic)) Cls2_te.y_logic];
            CNT_te.class=[Cls1_te.class;Cls2_te.class];
            
            % CV_verification
            CV_vrf_CNT=0;
            for ii=1:length(CNT_te.t)
                for jj=1:length(CNT_tr.t)
                    if CNT_te.t(ii)==CNT_tr.t(jj)
                        CV_vrf_CNT=CV_vrf_CNT+1;
                    end
                end
            end
            
            %% Parameter
             % channel index
            % Ch. Selection
            % 118 channels (all)
            %         channel_index=1:118;
            % 19 channels (M1)
            channel_index=[33,34,35,36,37,38,39,...
                           51,52,53,54,55,56,57,...
                           69,70,71,72,73,74,75]; % (FC1~6,C1~6,Cz,CP1~6) 21개
            %                 channel_index=[51,57, 69,75];
            %                 channel_index=[33,39, 51,57, 69,75];
            %         8 channels (Ear)
            %         channel_index=[31,50,68,67, 41,58,76,77];
            %                     channel_index=[31,41,50,58  67,68,76,77];
            %                     channel_index=[             67,68,76,77];
            %                     channel_index=[31,41,50,58 ];
            %                     channel_index=[31,41,50,58  67,68,76,77];
            
            
            % 8 (ear) + 21 (M1)
%             channel_index=[31,50,68,67, 41,58,76,77,...
%                            33,34,35,36,37,38,39,...
%                            51,52,53,54,55,56,57,...
%                            69,70,71,72,73,74,75]; % (FC1~6,C1~6,Cz,CP1~6) 19개
            band=[8 30];
            interval=[350 2490];
            num_patt=2;
%             t_lag=100;
            %% Training
            % channel selection
            CNT_tr=prep_selectChannels(CNT_tr,{'Index',channel_index});
            % BPF
            CNT_tr=prep_filter(CNT_tr, {'frequency', band});
            % Segmentation
            SMT_tr=prep_segmentation(CNT_tr, {'interval', interval});
            % Spatial filter
            [SMT_tr, CSP_W, CSP_D]=func_csp(SMT_tr,{'nPatterns', num_patt});
            % logvar
            FT_tr=func_featureExtraction(SMT_tr, {'feature','logvar'});
            % Feature selection
%             FT_tr=func_featureSelection(FT_tr);
           % Classifier
            [CF_PARAM]=func_train(FT_tr,{'classifier','LDA'});
            
            %% Test
            % channel selection
            CNT_te=prep_selectChannels(CNT_te,{'Index',channel_index});
            % BPF
            CNT_te=prep_filter(CNT_te, {'frequency', band});
            % Segmentation
            SMT_te=prep_segmentation(CNT_te, {'interval', interval});
            % Projection
            SMT_te=func_projection(SMT_te, CSP_W);
            % logvar
            FT_te=func_featureExtraction(SMT_te, {'feature','logvar'});
            % Feature selection
%             FT_te=func_featureSelection(FT_te);            
            % Classification
            [cf_out]=func_predict(FT_te, CF_PARAM);
            
            %% Evaluation
            N=length(cf_out);
            cor=0;
            
            for ii=1:N
                if (ii<N/2+1&&cf_out(ii)<0) % 이거 이렇게 해도 되나?
                    cor=cor+1;
                end
                if (ii>N/2&&cf_out(ii)>0)
                    cor=cor+1;
                end
            end
            acc(qq)=cor/N*100;
        end
        accsub(Subj,tt)=mean(acc);
    end
    Accuracy=mean(accsub,2);
end
mean(acc)

% accsub=accsub'
% mean(accsub)
%% Analysis

% scatter3(FT_tr.x(1,1:40),FT_tr.x(2,1:40),FT_tr.x(3,1:40))
% hold on
% scatter3(FT_tr.x(1,41:80),FT_tr.x(2,41:80),FT_tr.x(3,41:80))
% hold on
% scatter3(FT_te.x(1,1:10),FT_te.x(2,1:10),FT_te.x(3,1:10))
% hold on
% scatter3(FT_te.x(1,11:20),FT_te.x(2,11:20),FT_te.x(3,11:20))
% close all

% hold on
% scatter(FT_tr.x(1,1:40),FT_tr.x(2,1:40),'r.')
% hold on
% scatter(FT_tr.x(1,41:80),FT_tr.x(2,41:80),'b.')
% axis([-2 1 -2 1]);
% hold on
% scatter(FT_te.x(1,1:10),FT_te.x(2,1:10))
% hold on
% scatter(FT_te.x(1,11:20),FT_te.x(2,11:20))


