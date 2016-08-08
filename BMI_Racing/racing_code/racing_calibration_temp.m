function [CF_PARAM] = racing_calibration_temp( file ,band, fs, varargin)
%CAL_LOSS Summary of this function goes here
%   Detailed explanation goes here
opt = opt_cellToStruct(varargin{:});
marker={'1','left';'2','right';'3','foot'; '4', 'rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', fs});
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
mCNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
mCNT=prep_filter(mCNT, {'frequency', band});

%% binary classifier
set{1}={'binary';{'right', 'left'}};
set{2}={'binary';{'right', 'foot'}};
set{3}={'binary';{'left', 'foot'}};
set{4}={'binary';{'right', 'rest'}};
set{5}={'binary';{'left', 'rest'}};
set{6}={'binary';{'foot', 'rest'}};
set{7}={'ovr'; {'right', 'others'}};
set{8}={'ovr'; {'left', 'others'}};
set{9}={'ovr'; {'foot', 'others'}};
set{10}={'ovr'; {'rest', 'others'}};

out=changeLabels(mCNT,{'right',1;'others',2});
SMT=prep_segmentation(mCNT, {'interval', [750 3500]});
x=squeeze(log(var(SMT.x)))
x=x'
n_tr=100;
for j=1:length(SMT.class)
    for i=1:length(SMT.chan)
        [a b]=find(mCNT.y_dec==j)
        mn(j, i)=mean(x(i,b));
        sd(j, i)=std(x(i,b));
    end
end

for j=1:length(SMT.class)
    for i=1:length(SMT.chan)
        a_feature{j,i}=normrnd(mn(j,i),sd(j, i),[1 n_tr]);
    end
end

tm=[]
for j=1:length(SMT.class)
    for i=1:length(SMT.chan);
       tm =cat(1,tm,a_feature{j,i});
    end
    a_feat{j}=tm
    tm=[];
end

tm=[];
for j=1:length(SMT.class)
tm=cat(2, tm,a_feat{j})
end

set2={
    [1:100 101:200]; 
    [1:100 201:300]; 
    [101:200 201:300]; 
    [1:100 301:400] ;
    [101:200 301:400] ; 
    [201:300 301:400] ;
    [1:100 101:300] ;
    [101:200 1:100 201:300] ;
    [201:300 1:200] ;
    [301:400 1:300]
    }

fv.x=tm;
fv.y(1,1:100)=1;
fv.y(2,101:200)=1;
fv.y(3,201:300)=1;
fv.y(4,301:400)=1;

for i=1:length(set2)
    FT.x=fv.x(:,set2{i})
    
    if i>=1 && i<=6
        FT.y_logic(1,1:100)=1;
        FT.y_logic(2,101:200)=1;
    elseif i>=7 && i<=9
        FT.y_logic(1,1:100)=1;
        FT.y_logic(2,101:300)=1;
    elseif i==10
        FT.y_logic(1,1:100)=1;
        FT.y_logic(2,101:400)=1;
    end
[CF_PARAM{i}]=func_train(FT,{'classifier','LDA'});
clear FT
end
% 
% file2='C:\Vision\Raw Files\20160728_mhlee_short'
% [EEG.data, EEG.marker, EEG.info]=Load_EEG(file2,{'device','brainVision';'marker', marker;'fs', fs});
% field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
% mCNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% mCNT=prep_filter(mCNT, {'frequency', band});
% 
% for i=1:length(set)
%     if strcmp(set{1,i}{1,1},'binary')
%         CNT=prep_selectClass(mCNT,{'class',{set{1,i}{2,1}{:}}});
%     else %
%         if i==7 || i==8 || i==9 % ovr between active classes - delete rest class
%             CNT=prep_removeClass(mCNT, 'rest');
%             CNT=changeLabels(CNT,{set{1,i}{2,1}{1},1;'others',2});
%         else % rest vs all others
%             CNT=changeLabels(mCNT,{set{1,i}{2,1}{1},1;'others',2});
%         end
%     end
%     SMT=prep_segmentation(CNT, {'interval', [750 3500]});
% 
%     FT=func_featureExtraction(SMT, {'feature','logvar'});
% 
%  [cf_out]=func_predict(FT, CF_PARAM{i})
%     
%     CV.var.band=band;
%     CV.var.interval=[750 3500];
%     CV.prep={ % commoly applied to training and test data before data split
%         'CNT=prep_filter(CNT, {"frequency", band})'
%         'SMT=prep_segmentation(CNT, {"interval", interval})'
%         };
%     CV.train={
%         '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
%         'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%         '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
%         };
%     CV.test={
%         'SMT=func_projection(SMT, CSP_W)'
%         'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%         '[cf_out]=func_predict(FT, CF_PARAM)'
%         };
%     CV.option={
%         'KFold','5'
%         % 'leaveout'
%         };
%     
%     [loss{i}]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo
% end

end

