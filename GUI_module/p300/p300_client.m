function p300_client( clf_param, varargin )
opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'segTime'),segTime=[-200 800];else segTime=opt.segTime;end
if ~isfield(opt,'baseTime'),baseTime=[-200 0];else baseTime=opt.baseTime;end
if ~isfield(opt,'selTime'),selTime=[0 800];else selTime=opt.selTime;end
if ~isfield(opt,'nFeature'),nFeature=10;else nFeature=opt.nFeature;end
if ~isfield(opt,'nSequence'),nSequence=10;else nSequence=opt.nSequence;end
channel_idx=opt.channel;


%% variables
load('cell_order_new.mat')

%% server-client
d = tcpip('localhost', 3000, 'NetworkRole', 'Client');
set(d, 'OutputBufferSize', 1024); % Set size of receiving buffer, if needed.
while(1)
    try
        fopen(d);
        break;
    catch
        fprintf('%s \n','Cant find Server');
    end
end
connectionSend = d;

%% ONLINE-SPELLER
spell_char = ['A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'];



EEG_data=[];
% tic;
bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
i=0; j=1; nChar=1; nc=1; in_nc=1; nSeq=1;


escapeKey = KbName('esc');

t=[];IV=[];
idc= floor(segTime(1)*state.fs/1000):ceil(segTime(2)*state.fs/1000);
T= length(idc);
nEvents= 1;

tic

DAT=cell(1,36);
% tm_Dat=zeros(36,nDat*nChans);  % 6x6, length of feature dimension
tm_Dat=zeros(36,length(clf_param.cf_param.w));
run_p300=true;
run=true;
while run
    [ keyIsDown, seconds, keyCode ] = KbCheck;
    if keyIsDown
        if keyCode(escapeKey)
            run=false;
        end
    end
    
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    data=data(:,channel_idx);           % channel 부분 추가
    EEG_data = [EEG_data; data];        % ★ data 앞부분 잘라내기 필요
    if ~isempty(markerdescr)
    end
    if ~isempty(markerdescr)
        if length(markerdescr)>1
            markerdescr=markerdescr(1);
        end
        
        if markerdescr >= 1 && markerdescr <=12
            run_p300=true;
            i=i+1;
            t(i)=toc;
            marker(i)=markerdescr;
        else markerdescr>20             % ???
            run_p300=false;
            
        end
        
        if markerdescr==14
            fwrite(d, b);
            t=[];
            i=0;
            nc=1;
            in_nc=1;
            nChar=nChar+1;
            nSeq=1;
            DAT=cell(1,36); % init.
            tm_Dat=zeros(36,length(clf_param.cf_param.w));
            run_p300=false;
        end
        if markerdescr==20
            %             save([fold '\subject_log\' FILE FILE2 '_MATLAB2_DATA'], 'SAVE_DATA')
%                         bbci_acquire_bv('close');
            break;
        end
        
    end
    
    if run_p300
        if nc <= i
            if ~isempty(t)
                [nDat nChans]=size(EEG_data);
                mrk.t=t(nc)*state.fs;  %% nc 수정
                IV= round(idc(:)*ones(1,nEvents) + ones(T,1)*mrk.t);
            end
            if ~isempty(IV)
                if IV(end)<length(EEG_data)
                    dat.x= reshape(EEG_data(IV, :), [T, nEvents, nChans]);
                    dat.t= linspace(segTime(1), segTime(2), length(idc));
                    dat.fs=state.fs;
                    dat.ival=segTime(1):1000/state.fs:segTime(end);
                    dat=prep_baseline(dat, {'Time',baseTime});
                    dat.chan=state.clab(channel_idx);
                    dat=prep_selectTime(dat, {'Time',selTime});
                    dat=func_featureExtraction(dat,{'feature','erpmean';'nMeans',nFeature});
                    
                    [nDat, nTrials, nChans]= size(dat.x);
                    dat.x= reshape(permute(dat.x,[1 3 2]), [nDat*nChans nTrials]);
                    
                    for i2=1:6
                        DAT{cell_order{nSeq}(in_nc,i2)}(end+1,:)=dat.x;
                    end
                    for i2=1:length(spell_char)
                        if size(DAT{i2},1)==1
                            tm_Dat(i2,:)=DAT{i2};
                        else
                            tm_Dat(i2,:)=mean(DAT{i2});
                        end
                    end
                    [Y]=func_predict(tm_Dat', clf_param);
                    [a b]=sort(Y, 'ascend');
                    spell_char(b)
                    
                    sprintf('nc: %d, in_nc: %d n_seq', nc, in_nc, nSeq)
                    nc=nc+1;
                    in_nc=in_nc+1;
                    if in_nc>12
                        in_nc=1;
                        nSeq=nSeq+1;
                    end
                end
            end
        end
    end
end
% save('online_dat.mat', 'dat_seq_all');