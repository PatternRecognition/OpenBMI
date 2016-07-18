function [ dat, Lap ] = prep_laplacian( data, varargin )
% prep_laplacian: 
%  This function recalculates the signal of selected channels by
%  subtracting the average value of a set of surrounding electrodes.
%  [Tandonnet et al., 2005]
%    - Small Laplacian: Using electrodes within 3cm from the selected channel
%    - Large Laplacian: Using electrodes within 6cm from the selected channel
%
% Example:
%  [out] = prep_laplacian(data,{'Channel',{'C1','Cz','C2'}})
%
% Input:
%   data - Data structure, segmented or continuous EEG data
% 
% Option:
%   Channel    - selected channels to apply Laplacian filter 
%   filterType - 'small'(default) or 'large'
%
% Output:
%   dat - Filtered data using Laplacian filter in selected channels
%   Lap - only filtered data and channel
% 
%
% Reference:
%   C. Tandonnet, B. Burle, T. Hasbroucq, and F. Vidal, "Spatial Enhancement of 
%   EEG Traces by Surface Laplacian Estimation: Comparison between Local and Global methods," 
%   Clinical Neurophysiology, Vol. 116, No. 1, 2005, pp. 18-24.
%
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
%

dat = data;
opt = opt_cellToStruct(varargin{:});

channel = opt_channelMontage(varargin);

if ~isfield(opt,'filterType')
    warning('Set the default');
    opt.filterType = 'small';
end
% 들어온 채널을 channel orgin 에서의 인덱스 알기

for numChannel = 1: size(opt.Channel,2)
    for chInxFindRow = 1 : size(channel.origin,1)
        for chInxFindCol = 1 : size(channel.origin,2)
            if strcmp(opt.Channel{numChannel}, channel.origin{chInxFindRow,chInxFindCol}) ==1
                chInx{numChannel} = [chInxFindRow chInxFindCol];
            end
        end
    end
end

% Laplacian 되는 지 확인
switch(opt.filterType)
    case 'small'
        for numChInx = 1: size(chInx,2)
            if chInx{numChInx}(1) == 9 || chInx{numChInx}(2) == 11
                warning('Do not apply small laplacian filter , because of "%s" \n' , channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                warning('Exceed the channel of index, Please check the channel index \n');
                
            else
                checkCh = [channel.label{chInx{numChInx}(1)-1,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)-1};...
                    channel.label{chInx{numChInx}(1)+1,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)+1};];
                if sum(checkCh) == 4
                    fprintf('Possible to apply small laplacian filter in %s \n', channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                    
                else
                    warning('Do not apply small Laplacian filter in selected channel \n');
                end
            end
        end
    case 'large'
        for numChInx = 1: size(chInx,2)
            if chInx{numChInx}(1) == 8 ||chInx{numChInx}(1) == 9 || chInx{numChInx}(2) == 10 || chInx{numChInx}(2) == 11
                warning('Do not apply large laplacian filter , because of %s \n' , channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                warning('Exceed the channel of index, Please check the channel index');
            else
                checkCh = [channel.label{chInx{numChInx}(1)-2,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)-2};...
                    channel.label{chInx{numChInx}(1)+2,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)+2};];
                if sum(checkCh) == 4
                    fprintf('Possible to apply large laplacian filter in %s \n' , channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                else
                    warning('Do not apply small Laplacian filter in selected channel \n');
                end
            end
        end
end
%% Calulate laplacian filter
% 실제데이터 레이블 찾기

Lap.x = cell(1,size(chInx,2));
for num =1: size(chInx,2)
    Lap.clab{num} = channel.origin{chInx{num}(1),chInx{num}(2)};
    
end
switch(opt.filterType)
    case 'small'
        for numChInx = 1: size(chInx,2)
            for dataCh = 1:size(dat.chan,2)
                if strcmp(channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)},dat.chan{dataCh})==1
                    dataLabel{numChInx}= dataCh;
                end
                if strcmp(channel.origin{chInx{numChInx}(1)-1,chInx{numChInx}(2)},dat.chan{dataCh})==1
                    LapdataLabelTop = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1)+1,chInx{numChInx}(2)},dat.chan{dataCh})==1
                    LapdataLabelBottom = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)-1},dat.chan{dataCh})==1
                    LapdataLabelLeft = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)+1},dat.chan{dataCh})==1
                    LapdataLabelRight = dataCh;
                end
            end
            % Trial별로 해당 채널 Laplacian 필터한 값
            if ndims(dat.x) ==2
                avgLap = (dat.x(:,LapdataLabelBottom)+dat.x(:,LapdataLabelLeft)+dat.x(:,LapdataLabelRight)+dat.x(:,LapdataLabelTop))/4;
                
                LapData = dat.x(:,dataLabel{numChInx})-avgLap;
                
                Lap.x{numChInx} = LapData;
                dat.x(:,dataLabel{numChInx}) = Lap.x{numChInx};
            else
                for trial = 1: size(dat.x,2)
                    avgLap(:,trial) = (dat.x(:,trial,LapdataLabelBottom)+dat.x(:,trial,LapdataLabelLeft)+dat.x(:,trial,LapdataLabelRight)+dat.x(:,trial,LapdataLabelTop))/4;
                    
                    LapData(:,trial) = dat.x(:,trial,dataLabel{numChInx})-avgLap(:,trial);
                end
                Lap.x{numChInx} = LapData;
                dat.x(:,:,dataLabel{numChInx}) = Lap.x{numChInx};
            end
            
        end
        
    case 'large'
        for numChInx = 1: size(chInx,2)
            for dataCh = 1:size(dat.chan,2)
                if strcmp(channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)},dat.chan{dataCh})==1
                    dataLabel= dataCh;
                end
                if strcmp(channel.origin{chInx{numChInx}(1)-2,chInx{numChInx}(2)},dat.chan{dataCh})==1
                    LapdataLabelTop = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1)+2,chInx{numChInx}(2)},dat.chan{dataCh})==1
                    LapdataLabelBottom = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)-2},dat.chan{dataCh})==1
                    LapdataLabelLeft = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)+2},dat.chan{dataCh})==1
                    LapdataLabelRight = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1)-1,chInx{numChInx}(2)-1},dat.chan{dataCh})==1
                    LapdataLabelMedLeft = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1)-1,chInx{numChInx}(2)+1},dat.chan{dataCh})==1
                    LapdataLabelMedRight = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1)+1,chInx{numChInx}(2)-1},dat.chan{dataCh})==1
                    LapdataLabelDownLeft = dataCh;
                elseif strcmp(channel.origin{chInx{numChInx}(1)+1,chInx{numChInx}(2)+1},dat.chan{dataCh})==1
                    LapdataLabelDownRight = dataCh;
                end
            end
            if ndims(dat.x) ==2
                avgLap = (dat.x(:,LapdataLabelBottom)+dat.x(:,LapdataLabelLeft)+dat.x(:,LapdataLabelRight)+dat.x(:,LapdataLabelTop)+ ...
                    dat.x(:,LapdataLabelMedLeft)+dat.x(:,LapdataLabelMedRight)+dat.x(:,LapdataLabelDownLeft)+dat.x(:,LapdataLabelDownRight))/8;
                
                LapData = dat.x(:,dataLabel{numChInx})-avgLap;
                Lap.x{numChInx} = LapData;
                dat.x(:,dataLabel{numChInx}) = Lap.x{numChInx};
            else
                
                for trial = 1: size(dat.x,2)
                    avgLap(:,trial) = (dat.x(:,trial,LapdataLabelBottom)+dat.x(:,trial,LapdataLabelLeft)+dat.x(:,trial,LapdataLabelRight)+dat.x(:,trial,LapdataLabelTop)+ ...
                        dat.x(:,trial,LapdataLabelMedLeft)+dat.x(:,trial,LapdataLabelMedRight)+dat.x(:,trial,LapdataLabelDownLeft)+dat.x(:,trial,LapdataLabelDownRight))/8;
                    
                    LapData(:,trial) = dat.x(:,trial,dataLabel{numChInx})-avgLap(:,trial);
                end
                Lap.x{numChInx} = LapData;
                dat.x(:,:,dataLabel{numChInx}) = Lap.x{numChInx};
            end
        end
end



end

