%% Determine latency of switching the frequency and of the function fullscreen()

VP_CODE='latency'
acq_makeDataFolder;

if isempty(TODAY_DIR)
  error('Define TODAY_DIR!');
end
record=0 % 1: amplifier connected, record EEG data, 0: testing
setup_sony3d_flicker2
[Click1, Fs, nbits, readinfo] = wavread('Click1');
Click1=Click1(1:1000)
stimulus=imread('jungle_normal','jpg');

%% a) Determine latency with photodiode and oscilloscope: marker set <--> frequency switched

if record,  bvr_startrecording(['sony3d_' VP_CODE], 'impedances', 0); pause(3);end

fprintf('Press enter to start\n'); pause

for freq=2*(39:2:99)
soundsc(Click1,Fs),
pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, freq), 
ppTrigger(freq);
display(freq/2),pause(4),
end

pause(1)
if record,  bvr_sendcommand('stoprecording'); end
pause(1)

%% Similar, but always jump back to 78Hz
if record,  bvr_startrecording(['sony3d_' VP_CODE '_backto78Hz'], 'impedances', 0); pause(3);end

fprintf('Press enter to start\n'); pause
freqs=[repmat(2*39,1,30) ; 2*(41:2:99)];
freqs=freqs(:);
for i=1:length(freqs)
  freq=freqs(i);
  soundsc(Click1,Fs)
  pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, freq),
  ppTrigger(freq);
  display(freq/2),pause(4),
end

pause(1)
if record,  bvr_sendcommand('stoprecording'); end
pause(1)

%% b) Determine latency with photodiode and gTRIGbox: marker set <--> fullscreen(jungle) appears on the screen

if record,  bvr_startrecording(['sony3d_' VP_CODE '_fullscreen'], 'impedances', 0); pause(3);end

fprintf('Press enter to start\n'); pause

for i=1:50
  fullscreen(stimulus,2)
  ppTrigger(2);
  pause(3)
  closescreen()
  ppTrigger(3);
  display(i),pause(2),
end

pause(1)
if record,  bvr_sendcommand('stoprecording'); end
pause(1)

%% Close tcp/ip connection
pnet(tcp_conn, 'printf', 'freq %d %d %d\n', 800, 600 ,85);
pnet(tcp_conn, 'close');
fprintf('done!\n');