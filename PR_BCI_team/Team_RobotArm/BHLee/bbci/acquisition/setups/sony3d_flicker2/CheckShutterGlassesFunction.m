%% Check with oscilloscope (spectrum) if shutter glasses work properly
% Also check whether you percieve the switch

record=0; % 1: amplifier connected, record EEG data, 0: testing
VP_CODE='test';
acq_makeDataFolder;

% start tcp/ip connection
display('Server started?'),pause
setup_sony3d_flicker2
display('Turn off CRT!'),pause
[Click1, Fs, nbits, readinfo] = wavread('Click1');Click1=Click1(1:1000);

for freqs=2*(39:2:99)
  soundsc(Click1,Fs),
  pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, freqs),
  display(freqs/2),pause,
end


pnet(tcp_conn, 'printf', 'freq %d %d %d\n', 800, 600 ,85);
pnet(tcp_conn, 'close');
fprintf('done!\n'); 