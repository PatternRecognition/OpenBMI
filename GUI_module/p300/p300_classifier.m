function [clf_param,ch_idx]=p300_classifier(EEG,varargin)
opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'segTime'),segTime=[-200 800];else segTime=opt.segTime;end
if ~isfield(opt,'baseTime'),baseTime=[-200 0];else baseTime=opt.baseTime;end
if ~isfield(opt,'selTime'),selTime=[0 800];else selTime=opt.selTime;end
if ~isfield(opt,'nFeature'),nFeature=10;else nFeature=opt.nFeature;end

load cell_order_new

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);

if ~isfield(opt,'channel')
    ch_idx=1:length(cnt.chan);
else
    channel=channelFormat(opt.channel);
    ch_idx= find(ismember(cnt.chan,channel{:}));
end
cnt=prep_selectChannels(cnt,{'Index',ch_idx});

smt=prep_segmentation(cnt, {'interval', segTime});
smt=prep_baseline(smt, {'Time',baseTime});
smt=prep_selectTime(smt, {'Time',selTime});
fv=func_featureExtraction(smt,{'feature','erpmean';'nMeans',nFeature});

[nDat, nTrials, nChans]= size(fv.x);
fv.x= reshape(permute(fv.x,[1 3 2]), [nDat*nChans nTrials]);

[clf_param]=func_train(fv,{'classifier','LDA'});

end
function channel=channelFormat(channel)

ch=opt_getToken(channel,',');
channel=sprintf('{''%s''',ch{1});
if length(ch)>1
    for i=2:length(ch)
        channel=[channel,',''',ch{i},''''];
    end
end
channel=[channel,',}'];
channel=eval(sprintf('{%s}',channel));

end