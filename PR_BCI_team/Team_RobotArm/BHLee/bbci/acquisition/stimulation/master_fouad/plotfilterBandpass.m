function plotfilterBandpass(data_set_meantrial,data_set_meantrialSpec,cnt,t_trial)
 
% for i = 1:length(data_set_meantrialSpec),
%   
%   bands = data_set_meantrialSpec{2,i};
%  for ii = 1:length(bands)
%   figure
%   subplot(length(bands),1,ii)
%   plot(bands{2,ii})
% end
% end
% 
% 
% for ii = 1: length(data_set_meantrial),
%   figure
% band = data_set_meantrial{2,ii};
% 
% f =((0:length(band{2,1})-1)/length(band{2,1}))*cnt.fs; 
% 
%   for i = 1:length(band)
%     subplot(length(band),1,i)
%     plot(t_trial,band{2,i})
%   end
% end

for ii = 1: length(data_set_meantrialSpec),
  figure
band = data_set_meantrialSpec{2,ii};

f =((0:length(band{2,1})-1)/length(band{2,1}))*cnt.fs; 

  for i = 1:length(band)
    subplot(length(band),1,i)
    plot(f,band{2,i})
  end
end
