function [ output_args ] = p300_online(fig,txt, varargin)
%HYBRID_ONLINE Summary of this function goes here
%   Detailed explanation goes here
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (i) Offline trigger
%   '15' : Start trigger
%   '1' : Target trigger
%   '2' : Non-target trigger
%   '14' : 10-Sequence-end trigger
%   '20': End trigger
%
% (ii) Online trigger
%   '15' : Start trigger
%   '1~12' : Stimulus trigger
%   '14' : 10-Sequence-end trigger
%   '19' : Check - connection
%   '20': End trigger
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2017. 08. 03. Oyeon Kwon (oy_kwon@korea.ac.kr)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Init
opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'segTime'),segTime=[-200 800];else segTime=opt.segTime;end
if ~isfield(opt,'baseTime'),baseTime=[-200 0];else baseTime=opt.baseTime;end
if ~isfield(opt,'selTime'),selTime=[0 800];else selTime=opt.selTime;end
if ~isfield(opt,'nFeature'),nFeature=10;else nFeature=opt.nFeature;end
% if ~isfield(opt,'nSequence'),nSequence=10;else nSequence=opt.nSequence;end
if ~isfield(opt,'selectedFreq'),selectedFreq=[0.5 40];else selectedFreq=opt.selectedFreq;end

% check for connection
global sock
fwrite(sock,19);
global direct
direct=fileparts(which('OpenBMI'));

channel_idx=opt.channel;
CF_PARAM = opt.clf_param;
nFeatures=nFeature;

dat_seq=[];

spell_char = ['A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'];

temp2= {'A', 'B', 'C', 'D', 'E', 'F'; ...
    'G', 'H', 'I', 'J', 'K', 'L'; ...
    'M', 'N', 'O', 'P', 'Q', 'R'; ...
    'S', 'T', 'U', 'V', 'W', 'X'; ...
    'Y', 'Z', '1', '2', '3', '4'; ...
    '5', '6', '7', '8', '9', '_'};

EEG_data=[];
% tic
bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
i=0;j=1;nChar=1;nc=1; 
dat_seq_all=[]; % all sequence
p_1=zeros(1,3); % best for eog

Dat=zeros(size(CF_PARAM.cf_param.w,1), 12, 10); % calculate by input parameter: (feature size * channels) -- 2017.07.31 oyeon
eog_Key = KbName('1');
t=[];IV=[];count=0;
ival=segTime;
idc= floor(ival(1)*state.fs/1000):ceil(ival(2)*state.fs/1000);
T= length(idc);
nEvents= 1;
eog_tri=0;
t_c=0; % 한번만 작동하도록
tic;
N1=1;
in_nc=1;
nSeq=1;
DAT=cell(1,36); % for random speller
tm_Dat=zeros(36,size(CF_PARAM.cf_param.w,1)); % for random speller
run_p300=true;
% tic
load random_cell_order.mat;
while (true)
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    EEG_data = [EEG_data; data];
    if ~isempty(markerdescr)
    markerdescr;
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
           
        else markerdescr>20
            run_p300=false;
            
        end
        
        if markerdescr==14
            t_c=0;
            t=[];
            i=0;
            nc=1;
            nChar=nChar+1;
            in_nc=1;
            nSeq=1;
            DAT=cell(1,36); % init
            tm_Dat=zeros(36,size(CF_PARAM.cf_param.w,1));
            j=1;
            Dat=zeros(size(CF_PARAM.cf_param.w,1), 12, 10);            
            run_p300=false;            
        end
        if markerdescr==20
%         save([pwd '\OpenBMI-master\log\SAVE_DATA'], 'SAVE_DATA');    
         save(fullfile(direct, 'log',sprintf('%d%02d%02d_%02d.%02d_p300.mat',c(1:5))),'SAVE_DATA');  
            break;
        end
        
    end
%% run    
    if run_p300
        if nc <= i
            if ~isempty(t)
                [nDat nChans]=size(EEG_data);
                mrk.t=t(nc)*100;  %% nc 수정
                IV= round(idc(:)*ones(1,nEvents) + ones(T,1)*mrk.t);
            end
            if ~isempty(IV)
                if IV(end)<length(EEG_data)
                    dat.x= reshape(EEG_data(IV, :), [T, nEvents, nChans]); 
                    dat.x =  dat.x(:,:,channel_idx); 
                    dat.t= linspace(ival(1), ival(2), length(idc));
                    dat.fs=state.fs;
                    dat.ival = ival; 
                    dat=prep_baseline(dat, {'Time',baseTime}); 
                    dat.x=dat.x(find(dat.t==0):end,:,:);  %% channel selection and check baseline up to when t=0 -- 2017.07.31 (oyeon)
                    dat=func_featureExtraction(dat,{'feature','erpmean';'nMeans',nFeatures}); %% revised 2017.07.28 (oyeon)
                    [nDat, nTrials, nCh]= size(dat.x);  
                    dat.x = reshape(permute(dat.x,[1 3 2]), [nDat*nCh nTrials]); % data reshape (features * ch, trials)-- 2017.07.31 (oyeon)
                    marker(nc); 
                    Dat(:,marker(nc),j)=dat.x;  
                    infor=['trial:' int2str(nChar),'  ' ,'sequence:' int2str(j),'  ', 'run:' int2str(nc)];                  
                    
                    for i2=1:6
                        DAT{cell_order{nSeq}(in_nc,i2)}(end+1,:)=dat.x;
                    end                    
                    for i2=1:36
                        if size(DAT{i2},1)==1
                             tm_Dat(i2,:)=DAT{i2};
                        else
                        tm_Dat(i2,:)=mean(DAT{i2});
                        end
                    end
                    
                    [Y]=func_predict(tm_Dat', CF_PARAM);
                    [a b]=sort(Y, 'ascend');
                    y22=reshape(Y, 6, 6); 
                    y2=y22';
%                          YY{nSeq} = Y;          
%% Visualization and Find index
                    visualization_P300(fig,y2);
                    set(txt, 'string', infor);    drawnow;                                     
                    [maxNum, maxIndex] = min(y2(:));  %save best for eog
                    [row1, col1] = ind2sub(size(y2), maxIndex);
                    for ii=1  %-- just get the first target char (1:3 you will take to the third low target)
                        [maxNum, maxIndex] = min(y2(:));
                        [row, col] = ind2sub(size(y2), maxIndex);
                        p_1(ii)=[find(spell_char==temp2{row,col})];
                    end                    
%% Save                   
                    if nc==12*j
                        if j==10
                            SAVE_DATA{nChar}.dat=DAT;
                            SAVE_DATA{nChar}.nc=nc;
                            SAVE_DATA{nChar}.j=j;                               

                            fileID2 = fopen([direct '\log\_char.txt'],'w'); % selected  char matlab 보내기
                            t_char=temp2{row1,col1};
                            fprintf(fileID2,'%c\n',t_char);
                            fclose(fileID2);
                        else
                            j=j+1;
                        end
                    end
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
toc;
end
