function [ CARfilter ] = prep_commonAverageReference( data, varargin )
% prep_commonAverageReference: Subtracting in the average value of the entire electrode
% montage (the common average) from the interested channel [D. J. McFarland et al., 1997]
%
% Synopsis:
%  [filterData] = prep_commonAverageReference(data , <OPT>)
%
% Arguments:
%   data: Data structrue (ex) Epoched data or EEG raw data
%   <OPT> : 
%      .Channel - select the channel applied CAR filter 
%                 (e.g. {'Channel', {'C1', Cz', 'C2'}})
%      .filterType - exptChan: filtering except for the selected channel
%                  - incChan:  filtering include in the selected channel
%
% Return:
%    filterData:  Filtered data using common averge reference in selected channel
%
% See also:
%    opt_cellToStruct
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
    opt.filterType = 'exptChan';
end

all_chSum = 0;
CARfilter.x = dat.x;
for numChannel = 1: size(opt.Channel,2)
    for ch = 1: size(dat.chSet,2)
        if isequal(dat.chSet(ch), cellstr(opt.Channel{numChannel}) )
            channelIndex{numChannel}=ch;
            SelectChan{numChannel} = dat.x(:,:,channelIndex{numChannel});
        end
        all_chSum = dat.x(:,:,ch)+all_chSum;
    end
    
    switch opt.filterType
        case 'exptChan'
            CAR_filterData = SelectChan{numChannel} - ((all_chSum - SelectChan{numChannel})/size(dat.chSet,2));         
        case 'incChan'
            CAR_filterData = SelectChan{numChannel} - (all_chSum/size(dat.chSet,2));
    end
    CARfilter.data{numChannel} = CAR_filterData;
    CARfilter.clab{numChannel} = dat.chSet{channelIndex{numChannel}};
    CARfilter.x(:,:,channelIndex{numChannel}) = CARfilter.data{numChannel};
end


end



