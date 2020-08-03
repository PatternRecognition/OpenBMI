% Duration of the experiment depending on the number of frequencies and
% repetitions
duration=zeros(20,20)
for num_freqs = 1:20
  for num_repetitions  = 1:20
    pause_before_pic = 2; % [s]
    pause_during_pic = 10; % [s]
    estimated_time_to_answer = 2; %[s]
    estimated_time_between_runs = 15; %[s]
    duration(num_freqs,num_repetitions) = (num_repetitions * estimated_time_between_runs) + num_repetitions * num_freqs * (pause_before_pic + pause_during_pic + 2);
  end
end
figure, imagesc(duration./60),colorbar
xlabel('Repetitions'),ylabel('Number of frequencies')
title('Estimated duration in minutes')


% Frequencies
freqs=44:4:80;
figure,plot(freqs,'d'),grid on
title({['Frequencies: ' num2str(freqs)]; ['Number: ' num2str(numel(freqs))]})