function [ eeg ] = prep_filter( eeg, varargin )
%PROC_FILTER Summary of this function goes here
%   Detailed explanation goes here
if ~varargin{end}
    varargin=varargin{1,1}; %cross-validation procedures
end;

if length(varargin)>1; param=opt_proplistToCell(varargin{:});end

switch isstruct(eeg)
    case true %struct
        if isfield(eeg, 'cnt')
            tDat=eeg.cnt;
            eeg.cnt_old=eeg.cnt;
            fld='cnt';
        else % in case eeg.x
            tDat=eeg.x;
            eeg.x_old=eeg.x;
            fld='x';
        end
        
        if length(varargin)<2  %default parameter
            band=varargin{1};
            [b,a]= butter(5, band/eeg.fs*2,'bandpass');
            tDat(:,:)=filter(b, a, tDat(:,:));
        else
            switch lower(param{1})
                case 'frequency'
                    band=param{2};
                    [b,a]= butter(5, band/eeg.fs*2,'bandpass');
                    tDat(:,:)=filter(b, a, tDat(:,:));
            end
        end
        eeg.(fld)=tDat;
        
        % History
        if isfield(eeg,'stack')
            c = mfilename('fullpath');
            c = strsplit(c,'\');
            eeg.stack{end+1}=c{end};
        end
    case false
        % add if dat is not struct
end

end



