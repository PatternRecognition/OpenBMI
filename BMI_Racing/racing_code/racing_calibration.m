function [CSP_W, CF_PARAM, loss] = racing_calibration( file ,band, fs, varargin)
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

for i=1:length(set)
    if strcmp(set{1,i}{1,1},'binary')
        CNT=prep_selectClass(mCNT,{'class',{set{1,i}{2,1}{:}}});
    else %
        if i==7 || i==8 || i==9 % ovr between active classes - delete rest class
            CNT=prep_removeClass(mCNT, 'rest');
            CNT=changeLabels(CNT,{set{1,i}{2,1}{1},1;'others',2});
        else % rest vs all others
            CNT=changeLabels(mCNT,{set{1,i}{2,1}{1},1;'others',2});
        end
    end
    SMT=prep_segmentation(CNT, {'interval', [750 3500]});
    % Visualization module
    if opt.visualization == 1
        SMT = prep_resample(SMT,100);
        figure(1)
        subplot(2,5,i)
        visuspect = racing_visual_spectrum(SMT , {'Xaxis' , 'Frequency'; 'Yaxis' , 'Channel'});
    end
    [SMT, CSP_W{i}, CSP_D]=func_csp(SMT,{'nPatterns', [3]});

    FT=func_featureExtraction(SMT, {'feature','logvar'});
    [CF_PARAM{i}]=func_train(FT,{'classifier','LDA'});
    
    CV.var.band=band;
    CV.var.interval=[750 3500];
    CV.prep={ % commoly applied to training and test data before data split
        'CNT=prep_filter(CNT, {"frequency", band})'
        'SMT=prep_segmentation(CNT, {"interval", interval})'
        };
    CV.train={
        '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
        'FT=func_featureExtraction(SMT, {"feature","logvar"})'
        '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
        };
    CV.test={
        'SMT=func_projection(SMT, CSP_W)'
        'FT=func_featureExtraction(SMT, {"feature","logvar"})'
        '[cf_out]=func_predict(FT, CF_PARAM)'
        };
    CV.option={
        'KFold','5'
        % 'leaveout'
        };
    
    [loss{i}]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo
end

end

