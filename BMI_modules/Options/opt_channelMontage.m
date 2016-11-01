function [ channel ] = opt_channelMontage( varargin )
% opt_channelMontage:
% 
% Description:
% 	This function gets positions of standard named electrodes.
% 
% Example:
%     channel = opt_channelMontage();
% 
% Output:
%     .origin - 9x11 cell with standard name of electrodes
%     .label  - 9x11 cell with value of 1 where electrodes are, 0 the rest
% 
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
% 

channel.origin = {               '','','','Fp1','','','', 'Fp2','','','';
                                '','','AF7', 'AF3','','','','AF4', 'AF8','','';
                          '','F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8','';
                  'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10';
                         '', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8','';    
                  'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10';
                          '','P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8','';
                              '','','','PO7', 'PO3', 'POz', 'PO4', 'PO8','','','';
                               '','','','PO9', 'O1', 'Oz', 'O2', 'PO10','','',''};

channel.label  = cell(9,11);  
% channel.label = zeros(9,11);

% Channel.orignin 레이블 설정
% 채널 이면 1, 아니면 0
for chOrginRow =1 : size(channel.origin ,1)
    for chOriginCol = 1: size(channel.origin,2)
        if length(channel.origin{chOrginRow,chOriginCol}) > 0
            channel.label{chOrginRow,chOriginCol} = 1;
        else
            channel.label{chOrginRow,chOriginCol} = 0;
        end
    end
end


end

