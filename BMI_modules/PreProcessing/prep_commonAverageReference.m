function [ CARfilter ] = prep_commonAverageReference( data, varargin )
% prep_commonAverageReference: 
% 
% Description:
%   Subtracting average value of the entire electrode (the common average) 
%   from the interested channel. [D. J. McFarland et al., 1997]
%
% Example:
%  [filteredData] = prep_commonAverageReference(data,{'Channel',{'C1',Cz','C2'}})
%
% Input:
%   data: Data structure, segmented or raw EEG data
% Option: 
%   Channel    - selected channels to apply CAR filter 
%   filterType - exptChan: filtering except for the selected channel
%                incChan : (default) filtering including the selected channel
% 
% Return:
%    filteredData - CAR filtered data in selected channel
%
% Reference:
%   D. J. McFarland, L. M. McCane, S. V. David, and J. R. Wolpaw, "Spatial Filter 
%   Selection for EEG-Based Communication," Electroencephalography and Clinical 
%   Neurophysiology, Vol. 103, No. 3, 1997, pp. 386-394.
%
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
%

dat = data;
opt = opt_cellToStruct(varargin{:});

if ~isfield(opt,'filterType')
    warning('Set the default');
    opt.filterType = 'incChan';
end

all_chSum = 0;
CARfilter.x = dat.x;
for numChannel = 1: size(opt.Channel,2)
    for ch = 1: size(dat.chan,2)
        if isequal(dat.chan(ch), cellstr(opt.Channel{numChannel}) )
            channelIndex{numChannel}=ch;
            SelectChan{numChannel} = dat.x(:,:,channelIndex{numChannel});
        end
        all_chSum = dat.x(:,:,ch)+all_chSum;
    end
    
    switch opt.filterType
        case 'exptChan'
            CAR_filterData = SelectChan{numChannel} - ((all_chSum - SelectChan{numChannel})/size(dat.chan,2));         
        case 'incChan'
            CAR_filterData = SelectChan{numChannel} - (all_chSum/size(dat.chan,2));
    end
    CARfilter.data{numChannel} = CAR_filterData;
    CARfilter.clab{numChannel} = dat.chan{channelIndex{numChannel}};
    CARfilter.x(:,:,channelIndex{numChannel}) = CARfilter.data{numChannel};
end


end



