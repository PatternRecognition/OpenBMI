function output = online_ssvep_Analysis(axes_fft, axes_cca, results_text,varargin)
opt.buffer_size = 5000; opt.data_size = 1500; opt.feedback_freq = 100/1000;
opt.channel = 58; opt.TCPIP = 'off';
% Description:
%
% Input:
%
% Output:
%
% Trigger Information:
% 111 start a paradigm
% 222 finish a paradigm
% 1-5 start a stimuli
% 123 finish a stimuli
% 19 check trigger
global state sock;

%% Connect to the paradigm
flushoutput(sock);
flushinput(sock);
for i = 1:5
    fwrite(sock,19);
    tmp = fread(sock,1);
    if i == 5
        output = 'Check your connection';
        return;
    elseif ~isempty(tmp)
        break;
    end
end
tot_trials = tmp;

opt = opt_cellToStruct(varargin{:});
class = opt.class;
freq = strsplit(num2str(opt.freq));
T = opt.time_stimulus;
fs = state.fs;
marker = vertcat(strsplit(num2str(1:size(opt.class,2))), opt.class)';
channel = opt.chan;
buffer_size=opt.buffer_size; % #point -> ms
data_size=opt.data_size;
feedback_t=opt.feedback_freq; % feedback frequency

Dat=zeros(buffer_size, size(channel,2));
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));
% orig_Dat=zeros(10000, size(channel,2));
SMT = zeros(fs * T+1, size(channel,2), tot_trials);
markerd = zeros(tot_trials,1);
escapeKey = KbName('esc');
waitKey=KbName('s');
isstart=false;
ischeck = false;
output = 'Unexpected Error';

init_window = 1/min(opt.freq) * 2 * fs;
update_vis = 0;
update_time = 0.2;
last_update = 0;

i = 0;
t = zeros(1,tot_trials);
window = T * fs; % 200 * 5 = 1000
while true
        [ keyIsDown, ~, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                return;
            elseif keyCode(waitKey)
                warning('stop')
                GetClicks([]);
            end
        end
    
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    if size(markerdescr,2) > 1,markerdescr = markerdescr(end); markertime=markertime(end); end %marker가 여러개 받아졌다면 마지막을 받아오도록... 문제발생할 수 있음...
    %% start 되기전까지 sync
    if ~isempty(markerdescr)
        switch markerdescr
            case 111 % 시작
                isstart = true;
                tic;
                orig_Dat = data(markertime:end,channel);
                chsize = size(orig_Dat,1);
                continue;
            case 222 % 종료
                data = data(:,channel);
                orig_Dat = [orig_Dat; data];
                SAVE_DATA.x = orig_Dat;
                SAVE_DATA.t = t;
                SAVE_DATA.rt = rt;
                SAVE_DATA.SMT =SMT;
                SAVE_DATA.mar = markerd;
                c = clock;
                direct=fileparts(which('OpenBMI'));
                save(fullfile(direct, 'log',sprintf('%d%02d%02d_%02d.%02d_ssvep.mat',c(1:5))),'SAVE_DATA');
                bbci_acquire_bv('close');
                fclose(sock);
                
                output = 'Finish';
                break;
            case 123 % 자극 끝
                if isstart
                    ischeck = true;
                end
            otherwise
                if markerdescr >= 1 && markerdescr <= 5
                    i = i+1;
                    t(i) = ceil(size(orig_Dat,1)+markertime/1000*fs);
                    markerd(i)=markerdescr;
                else
                    output = 'Pls check your trigger in bv';
                    return;
                end
        end
    elseif ~isstart
        continue;
    end
    %% Data 분석
    %채널 선택 추가하기
    data = data(:,channel);
    orig_Dat = [orig_Dat; data];
    
    if size(orig_Dat, 1) > init_window && update_vis > update_time * fs 
        % Starting visualization when collecting data at least 
        % 1/min(freq) * 2 seconds and over 200 ms
        if size(orig_Dat,1) > window + 1
            Dat = orig_Dat(end-window:end,:);
            time = T;
        else
            Dat = orig_Dat(:,:);
            time = (size(Dat,1)-1) / fs;
        end
        res_cca = ssvep_cca_analysis(Dat,{'marker',marker;'freq', opt.freq;'fs', fs;'time',time});
        visualization_SSVEP(axes_cca,{'results',res_cca;'marker',marker;'lim', [0 0.5]});
        res_fft = ssvep_fft_analysis(Dat,{'marker',marker;'freq', opt.freq;'fs', fs;'time',time});
        visualization_SSVEP(axes_fft,{'results',res_fft;'marker',marker;'lim', [0 30]});
        update_vis = 0;
        last_update = size(orig_Dat,1);
    else
        update_vis = size(orig_Dat,1) - last_update;
    end
    
    
    % 자극 끝났을때 결과 분석
    if ischeck
        if(size(orig_Dat(t(i):end,:),1) < window+1)
            disp('data acq...');
            continue;
        end
        
        Dat = orig_Dat(t(i):t(i)+window,:); % t=0 일떄부터
        SMT(:,:,i) = Dat;
        %         Dat = prep_filter(Dat, {'frequency', in.fRange; 'fs',fs});
        [~, ind_fft] = max(ssvep_fft_analysis(Dat,{'marker',marker;'freq', opt.freq;'fs', fs; 'time',T}));
        [~, ind_cca] = max(ssvep_cca_analysis(Dat,{'marker',marker;'freq', opt.freq;'fs', fs; 'time',T}));
        res = sprintf('FFT Resutls:\t%d\tCCA Resutls:\t%d\n%d/%d\tMarker:\t\t%d',...
            ind_fft,ind_cca, i, tot_trials, markerd(i));
        set(results_text,'String', {res, char(get(results_text,'String'))});
        ischeck = false;
        fwrite(sock,ind_cca);
    end
end