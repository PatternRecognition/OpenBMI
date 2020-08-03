function plot_Trial(trial,t_trial,mrk_pos,fs,num_freq)

trial_num = size(trial,2);
length_num_freq = length(num_freq);
for iii=0:(trial_num-1)

    mrk_pos_s103((iii+1),:) = mrk_pos.S103((iii*length_num_freq)+1:(iii*length_num_freq)+length_num_freq);

end

for i = 1:trial_num
mrk_pos_stim(i,:) = mrk_pos.Stim(:,i)-mrk_pos.S102(i);
mrk_pos_stim_s103(i,:) = mrk_pos_s103(i,:)-mrk_pos.S102(i);
subplot(trial_num,1,i)
plot(t_trial,trial(:,i))
grid on
ylim([(min(trial(:,i)-5)) (max(trial(:,i)+5))])
  text(0,max(trial(:,i)),' \bf \leftarrow S102','Rotation',90,'FontSize',7,'VerticalAlignment','middle')
  
  for ii = 1:length_num_freq
  text(mrk_pos_stim(i,ii)/fs,max(trial(:,i)),[' \bf \leftarrow',int2str(num_freq(ii))],'Rotation',90,'FontSize',7,'VerticalAlignment','middle','Color',[1 0 0])
  text(mrk_pos_stim_s103(i,ii)/fs,max(trial(:,i)),' \bf \leftarrow S103','Rotation',90,'FontSize',7,'VerticalAlignment','middle','Color',[0 0.5 0.8])  
  end
end
xlabel('Time [s]')
ylabel('Amplitude [mV]')
