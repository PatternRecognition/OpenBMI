function dat= proc_removeChannels(dat, varargin)
%dat= proc_removeChannels(dat, channel_list)
%
% IN  dat          - struct of continuous or epoched data
%     channel_list - list or cell array of strings holding the channel
%                    names of the channels that are to be removed

todel= chanind(dat, varargin{:});
dat.clab(todel)= [];
dat.x(:,todel,:)= [];
