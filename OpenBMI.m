function [ ] = OpenBMI( varargin )
%OPENBMI2 Summary of this function goes here
%   Detailed explanation goes here
if isempty(varargin) %find a openbmi directory when use this function in the inner directory
    FILE=[];
    CUR_FILE=strsplit(pwd,'\');
    for i=length(CUR_FILE):-1:1
        if  strfind(lower(CUR_FILE{i}),'openbmi');
            FILE=[];
            for j=1:i
                if j~=i
                    temp=strcat(CUR_FILE{j},'\');
                    FILE=strcat(FILE,temp)
                else
                    temp=CUR_FILE{j};
                    FILE=strcat(FILE,temp)
                end
                
            end
            break;
        else
            CUR_FILE{i}=[];
        end
    end
else
    FILE=[];
    FILE=varargin{1};
end
global BMI;
BMI.DIR=FILE;
BMI.EEG_DATA=[BMI.DIR '\BMI_data\DATA'];
BMI.CODE_DIR=[BMI.DIR '\BMI_modules'];
BMI.PARADIGM_DIR=[BMI.CODE_DIR '\Paradigms'] ;
BMI.IO_ADDR=hex2dec('C010');
BMI.IO_LIB=[BMI.CODE_DIR '\Options\Parallel_Port'];
% config_io;

if ischar(BMI.DIR)
addpath(genpath(BMI.DIR));
end
% cd(BMI.DIR);
end

