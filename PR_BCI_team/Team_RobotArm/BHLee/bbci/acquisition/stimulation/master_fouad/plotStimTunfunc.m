function [zero_mat_ref,sig_pau_ref,sig_pau,sig,sig_pau_trial, mod_freq_out] = plotStimTunfunc(Para)

t = [0:1/Para.fs:Para.act_time-1/Para.fs];
w_c = 2*pi*Para.carfreq;                % Carrier freq in rad/s

C = 1;                                  % Carrier amplitude
M = 1;                                  % Modulation amplitude

wave_carr = C * sin(w_c*t);             % carrier signal
num_freq = length(Para.modfreq);

num_trial = Para.num_trial;         % number of trials

ref_t = [0:1/Para.fs:Para.ref_time-1/Para.fs];              % reference interval i hvert trial

    % length of each trial in samples
 
length_pause = Para.ifi;
t_pause = [0:1/Para.fs:(length_pause)-(1/Para.fs)]; 

zero_mat_ref = zeros(num_trial,length(ref_t));
zero_mat_pau = zeros(num_freq,length(t_pause));

for iii = 1:Para.num_block,

for ii =1:Para.num_trial,
    
    indices = randperm(num_freq);     
    mod_freq = Para.modfreq(indices);   % The different modulation frequencies... 
    mod_freq_out(ii,:) = Para.modfreq(indices);                                   % ... are arranged in random order
    w_m = mod_freq*2*pi;                % Modulation frequencies in rad/s
  
    for i = 1:num_freq,  
            wave_mod(:,i) = M * sin(w_m(:,i)'*t);
            sig(i,:) = ( wave_mod(:,i)).*sin(w_c*t)';     % shift of frequency in each trial
            
   
    end
    sig_pau = horzcat(sig,zero_mat_pau);
    num_sig_pau = numel(sig_pau);
    
    sig_pau_trial(ii,:) = reshape(sig_pau',1,num_sig_pau);
  
end
 sig_pau_ref = horzcat(zero_mat_ref,sig_pau_trial);

end

%%
length_trial = ((Para.act_time*num_freq)+Para.ref_time)+num_freq*Para.ifi;    % length of each trial in sec
t_trial = [0:1/Para.fs:(length_trial)-(1/Para.fs)];     

figure(1)
for i = 1:(num_trial/2)

    subplot(num_trial/2,1,i,'align')
    plot(t_trial,sig_pau_ref(i,:))
%     title(['Trial',int2str(i)])
    ylabel(['Trial ',int2str(i)])
    xlim([0 length_trial])
end
% set(gcf,'OuterPosition',[4 300 2000 600]) 
set(gcf,'Units','centimeters')
set(gcf,'Position',[-2,0,37,20])
xlabel('Length [Sec]')

figure(2)
for ii = ((num_trial/2)+1):num_trial

    subplot((num_trial/2),1,ii-(num_trial/2),'align')
    plot(t_trial,sig_pau_ref(ii,:))
%      title(['Trial',int2str(ii)])
     ylabel(['Trial ',int2str(ii)])
    xlim([0 length_trial])
end
set(gcf,'Units','centimeters')
set(gcf,'Position',[-2,0,37,20])
xlabel('Length [Sec]')