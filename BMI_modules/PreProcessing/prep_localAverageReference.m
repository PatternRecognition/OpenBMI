function [ Lar ] = prep_localAverageReference( data, varargin )
% prep_localAverageReference (Pre-processing procedure):
% 
% Description:
%    Reference to local average; subtracting average value of all electrodes within a given radius 

% Example:
%     Lar = prep_localAverageReference(data, {'radius', '1'; 'channel' , 'Cz'});
% 
% Input:
%     data - 
% Option:
%     Channel   - selected channels to apply this function
%     radius    -  For a radius ofwithin a given boundary : 
%                  '1' is the neighborhood of Cz extends from C1 and C2
%                  '2' is the neighborhood of Cz extends from C3 and C4
% 
% Output:
%   Lar : Filtered data using Laplacian filter in selected channels
% 
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
%

dat = data;
opt = opt_cellToStruct(varargin{:});

channel = opt_channelMontage(varargin);

% Adapting channel montage
for numChannel = 1: size(opt.Channel,2)
    for chInxFindRow = 1 : size(channel.origin,1)
        for chInxFindCol = 1 : size(channel.origin,2)
            if strcmp(opt.Channel{numChannel}, channel.origin{chInxFindRow,chInxFindCol}) ==1
                chInx{numChannel} = [chInxFindRow chInxFindCol];
            end
        end
    end
end

% Check the input channel whether the Local Average Reference is applied or not
switch(opt.filterType)
    case '1'
        for numChInx = 1: size(chInx,2)
            if chInx{numChInx}(1) == 9 || chInx{numChInx}(2) == 11
                warning('Do not apply local average reference filter , because of "%s" \n' , channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                warning('Exceed the channel of index, Please check the channel index \n');
                
            else
                checkCh = [channel.label{chInx{numChInx}(1)-1,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)-1};...
                           channel.label{chInx{numChInx}(1)+1,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)+1};];
                if sum(checkCh) == 4
                    fprintf('Possible to apply local average reference filter in %s \n', channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                    
                else
                    warning('Do not apply local average reference filter in selected channel \n');
                end
            end
        end
    case '1.5'
        for numChInx = 1: size(chInx,2)
            if chInx{numChInx}(1) == 9 || chInx{numChInx}(2) == 11
                warning('Do not apply local average reference filter , because of "%s" \n' , channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                warning('Exceed the channel of index, Please check the channel index \n');
            else
                checkCh = [channel.label{chInx{numChInx}(1)-1,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)-1};...
                    channel.label{chInx{numChInx}(1)+1,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)+1};...
                    channel.label{chInx{numChInx}(1)+1,chInx{numChInx}(2)+1};channel.label{chInx{numChInx}(1)+1,chInx{numChInx}(2)-1};...
                    channel.label{chInx{numChInx}(1)-1,chInx{numChInx}(2)-1};channel.label{chInx{numChInx}(1)-1,chInx{numChInx}(2)+1};];
                
                if sum(checkCh) == 8
                    fprintf('Possible to apply local average reference filter in %s \n', channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                else
                    warning('Do not apply local average reference filter in selected channel \n');
                end
                
            end
        end
    case '2'
        for numChInx = 1: size(chInx,2)
            if chInx{numChInx}(1) == 8 ||chInx{numChInx}(1) == 9 || chInx{numChInx}(2) == 10 || chInx{numChInx}(2) == 11
                warning('Do not apply local average reference filter , because of %s \n' , channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                warning('Exceed the channel of index, Please check the channel index');
            else
                checkCh = [channel.label{chInx{numChInx}(1)-2,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)-2};...
                    channel.label{chInx{numChInx}(1)+2,chInx{numChInx}(2)};channel.label{chInx{numChInx}(1),chInx{numChInx}(2)+2};];
                
                if sum(checkCh) == 4
                    fprintf('Possible to apply local average reference filter in %s \n' , channel.origin{chInx{numChInx}(1),chInx{numChInx}(2)});
                else
                    warning('Do not apply local average reference filter in selected channel \n');
                end
            end
        end
end        

% Calculate the local average reference
% Sum of surround channel in the selected channel
basicSumLar = cell(1,size(chInx,2));
for findCh = 1: size(chInx,2)
    for dataCh = 1:size(dat.chSet,2)
        if strcmp(channel.origin{chInx{findCh}(1),chInx{findCh}(2)},dat.chSet{dataCh})==1
            dataLabel{findCh}= dataCh;
        end
        if strcmp(channel.origin{chInx{findCh}(1)-1,chInx{findCh}(2)},dat.chSet{dataCh})==1
            LardataLabelTop = dataCh;
        elseif strcmp(channel.origin{chInx{findCh}(1)+1,chInx{findCh}(2)},dat.chSet{dataCh})==1
            LardataLabelBottom = dataCh;
        elseif strcmp(channel.origin{chInx{findCh}(1),chInx{findCh}(2)-1},dat.chSet{dataCh})==1
            LardataLabelLeft = dataCh;
        elseif strcmp(channel.origin{chInx{findCh}(1),chInx{findCh}(2)+1},dat.chSet{dataCh})==1
            LardataLabelRight = dataCh;
        end
    end
    % Trial별로 해당 채널 Laplacian 필터한 값
    for trial = 1: size(dat.x,2)
        sumLar(:,trial) = dat.x(:,trial,LardataLabelBottom)+dat.x(:,trial,LardataLabelLeft)+dat.x(:,trial,LardataLabelRight)+dat.x(:,trial,LardataLabelTop);
    end
    basicSumLar{findCh} = sumLar;
end

% Calculate
Lar.x = cell(1,size(chInx,2));
for num =1: size(chInx,2)
    Lar.clab{num} = channel.origin{chInx{num}(1),chInx{num}(2)};
end
switch(opt.filterType)
    case '1'
        for larInx = 1: size(chInx,2)
            sumLarData = basicSumLar{larInx};
            for trial = 1: size(dat.x,2)
                avgLap(:,trial) = sumLarData(:,trial)/4;
                LarData(:,trial) = dat.x(:,trial,dataLabel{larInx})-avgLap(:,trial);
            end
            Lar.x{larInx} = LarData;
        end
    case '1.5'
        for larInx = 1: size(chInx,2)
            sumLarData = basicSumLar{larInx};
            for dataCh = 1:size(dat.chSet,2)

                if strcmp(channel.origin{chInx{larInx}(1)-1,chInx{larInx}(2)-1},dat.chSet{dataCh})==1
                    LardataLabelLeftUp = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1)-1,chInx{larInx}(2)+1},dat.chSet{dataCh})==1
                    LardataLabelRightUp = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1)+1,chInx{larInx}(2)-1},dat.chSet{dataCh})==1
                    LardataLabelLeftDown = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1)+1,chInx{larInx}(2)+1},dat.chSet{dataCh})==1
                    LardataLabelRightDown = dataCh;
                end
            end
            for trial = 1: size(dat.x,2)
                avgLap(:,trial) = (sumLarData(:,trial)+dat.x(:,trial,LardataLabelLeftUp)+dat.x(:,trial,LardataLabelRightUp)+dat.x(:,trial,LardataLabelLeftDown)+dat.x(:,trial,LardataLabelRightDown))/8;
                LarData(:,trial) = dat.x(:,trial,dataLabel{larInx})-avgLap(:,trial);
            end
            Lar.x{larInx} = LarData;
        end
    case '2'
        for larInx = 1: size(chInx,2)
            sumLarData = basicSumLar{larInx};
            for dataCh = 1:size(dat.chSet,2)
                
                if strcmp(channel.origin{chInx{larInx}(1)-1,chInx{larInx}(2)-1},dat.chSet{dataCh})==1
                    LardataLabelLeftUp = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1)-1,chInx{larInx}(2)+1},dat.chSet{dataCh})==1
                    LardataLabelRightUp = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1)+1,chInx{larInx}(2)-1},dat.chSet{dataCh})==1
                    LardataLabelLeftDown = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1)+1,chInx{larInx}(2)+1},dat.chSet{dataCh})==1
                    LardataLabelRightDown = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1)-2,chInx{larInx}(2)},dat.chSet{dataCh})==1
                    LardataLabelMaxTop = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1)+2,chInx{larInx}(2)},dat.chSet{dataCh})==1
                    LardataLabelMaxBottom = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1),chInx{larInx}(2)-2},dat.chSet{dataCh})==1
                    LardataLabelMaxLeft = dataCh;
                elseif strcmp(channel.origin{chInx{larInx}(1),chInx{larInx}(2)+2},dat.chSet{dataCh})==1
                    LardataLabelMaxRight = dataCh;
                end
            end
            for trial = 1: size(dat.x,2)
                avgLap(:,trial) = (sumLarData(:,trial)+dat.x(:,trial,LardataLabelLeftUp)+dat.x(:,trial,LardataLabelRightUp)+dat.x(:,trial,LardataLabelLeftDown)+dat.x(:,trial,LardataLabelRightDown)...
                                   +dat.x(:,trial,LardataLabelMaxTop)+dat.x(:,trial,LardataLabelMaxBottom)+dat.x(:,trial,LardataLabelMaxLeft)+dat.x(:,trial,LardataLabelMaxRight))/12;
                LarData(:,trial) = dat.x(:,trial,dataLabel{larInx})-avgLap(:,trial);
            end
            Lar.x{larInx} = LarData;
        end
        
end



end

