function dat= proc_bipolarEOGcustom(dat, chanList)
%
% function dat= proc_bipolarEOGcustom(dat, chanList)
%% computes manually specified EOG channels such as F9-F10 and EOGvu-Fp2
%
% INPUT
% dat - epo or cnt structure
%
% chanList:  either char (std2EOG/stdvEOG/stdhEOG) or matrix (n x 3) specifying the bipolar channels. The first
% column specifies the name of the resulting EOG channel, the 2 and 3.
% column specify the two channels used for bipolarity.
%
% example:
% cnt = proc_bipolarEOGcustom(cnt, {'EOGv','EOGvu','Fp2'; 'EOGh','F9','F10'})
% JohannesHoehne  04/2012

if ischar(chanList)
    switch chanList
        case 'std2EOG'
            chanList = {'EOGv','EOGvu','Fp2'; 'EOGh','F9','F10'}
        case 'stdvEOG'
            chanList = {'EOGv','EOGvu','Fp2'}
        case 'stdhEOG'
            chanList = {'EOGh','F9','F10'}
    end
end


for ii = 1:size(chanList,1)
    for jj=2:3
        if sum(strcmp(dat.clab, chanList{ii,jj})) ~= 1
            error('specified channel was not found')
        end
    end
end

bipolar_list = {};
for ii = 1:size(chanList,1)
    bipolar_list{ii} = sprintf('%s-%s', chanList{ii,2}, chanList{ii,3});
end


dat_EOG = proc_bipolarChannels(dat, bipolar_list);
dat_EOG.clab = chanList(:,1)';

dat = proc_appendChannels(dat, dat_EOG);

end