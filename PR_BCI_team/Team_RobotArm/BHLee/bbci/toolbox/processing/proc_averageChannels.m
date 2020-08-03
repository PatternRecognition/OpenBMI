function out= proc_averageChannels(dat, varargin)
%out= proc_averageChannels(dat, channel_list)
%
% IN  dat          - struct of continuous or epoched data
%     channel_list - list of cell arrays or strings, defining the channels to
%                    averaged.,
% EXAMPLE
%  out= proc_averageChannels(epo, 'PO7,8', 'O1,2', {'Fz','Pz'});

channel_list=varargin;

out= dat;
[T nChans, nEpos]= size(dat.x);
nChannel_Lists= length(channel_list);
out.x= zeros(T, nChannel_Lists, nEpos);
out.clab= cell(1, nChannel_Lists);
for ib= 1:nChannel_Lists,
  idx= chanind(dat, channel_list{ib});
  out.x(:,ib,:)= mean(dat.x(:,idx,:), 2);
  out.clab{ib}= vec2str(dat.clab(idx), '%s', '+');
end
