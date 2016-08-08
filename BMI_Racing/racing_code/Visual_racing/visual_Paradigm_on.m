function [trial, CSP, CLY_LDA] = visual_Paradigm_on(ch, ival, band, varargin )
%VISUAL_ERD_ON Summary of this function goes here
%   Detailed explanation goes here
% ch: ch index, ex. [1:13]
% ival: interval unit: ms, ex. [750 3500]

%%
if ~isempty(varargin)
    CLY_LDA=varargin{:};
else
    CLY_LDA=[];
end
if ~isempty(band)
    band=[11 20];
end
if ~isempty(ival)
    ival=[500 3500];
end


%% Client
d = tcpip('localhost', 3000, 'NetworkRole', 'Client');
set(d, 'OutputBufferSize', 1024); % Set size of receiving buffer, if needed.

%Trying to open a connection to the server.
while(1)
    try
        fopen(d);
        break;
    catch
        fprintf('%s \n','Cant find Server');
    end
end
connectionSend = d;

bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
orig_Dat=[];

buffer_size=5000;
data_size=1500;
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));

escapeKey = KbName('esc');
waitKey=KbName('s');
%% test
% fid = fopen('ny.txt','wt');


play=true;
mrk_start=false;
mrk_end=false;
tm_Dat=[];
num_t=1; % number of trials
cy_num=1;
% check this 10 trials x 3 class with resting class each

%  set={'rest', {'right', 'left', 'foot'}; ...
%      'right', {'left', 'foot'}; ...
%      'left', {'right', 'foot'};...
%      'foot', {'right', 'left'}
%      };
%
plotChannels = {'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4'};
freq = [7 13; 11 20];
bandPowers = [];
maxIter = 36;
tempFlag = true;

set_num={4, {1, 2, 3}; ...
    1, {2, 3}; ...
    2, {1, 3};...
    3, {1, 2}
    };
feedback_on=false;
update_cfy=[31 61 121 181 241 301 361 421];
% for the real-time output
buffer_size=5000;
data_size=1500;
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));
sending_on=false;
tic
while play
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    
    if ~isempty(markerdescr)
        if length(markerdescr)>1 % some error
            markerdescr=markerdescr(1);
            markerdescr;
        end
        
        if markerdescr==1 || markerdescr==2 || markerdescr==3 || markerdescr==4
            mrk_start=true;
            num_mrk=markerdescr;
            sending_on=true;
        elseif markerdescr==11 || markerdescr==22 || markerdescr==33 || markerdescr==44
            mrk_end=true;
            sending_on=false;
        elseif markerdescr==6
            break            
%             save('online_Dat', 'trial');
%             save('online_CSP', 'CSP');
%             save('online_LDA', 'CLY_LDA');
        end
    end
    
    if mrk_start
        tm_Dat=[tm_Dat; data];
        if mrk_end
            switch num_mrk
                case 1
                    trial{num_t,:}=tm_Dat;
                    idx(num_t)=num_mrk;
                case 2
                    trial{num_t,:}=tm_Dat;
                    idx(num_t)=num_mrk;
                case 3
                    trial{num_t,:}=tm_Dat;
                    idx(num_t)=num_mrk;
                case 4
                    trial{num_t,:}=tm_Dat;
                    idx(num_t)=num_mrk;
            end
            
            % Feature plotting
            if num_t == 1
                h = figure();
                set(h, 'position', [300 50 1400 900]);
                for p = 1 : length(plotChannels)
                    if mod(p, 3) ~= 0
                        x_gap = (mod(p, 3) - 1) * 0.33;
                    else
                        x_gap = 0.66;
                    end
                    h(p)= subplot(3, 3, p);
                    set(h(p), 'position', [(0.025 + x_gap) (0.025 + (1 - ceil(p / 3) * 0.33)) 0.3 0.275]);
                    title(h(p), plotChannels{p});
                    set(h,'Nextplot','add')
                    grid on;
                end
            end
            [h, bandPowers] = plot_onlineBand(h, trial{num_t}, num_mrk, freq, bandPowers, plotChannels, state, []);            
            
            tm_Dat=[]; % Initialization
            mrk_end=false;
            mrk_start=false;
            
            if num_t==300   % please setup the maximum number of trials, it could be changes by artifact rejection.
                play=false;
            end
            num_t=num_t+1
            cy_num=num_t; %prevent meaningless iteration
        end
    end
    
    %% initial classifier training
    if ismember(cy_num, update_cfy)
        cy_num=0; %prevent meaningless iteration
        for i=1:length(trial)
            tm_trial{i}=prep_filter(trial{i}, {'frequency', band; 'fs', state.fs});% check frequency
            tm_trial{i}=tm_trial{i}(ival(1)+1:ival(2),:); % !!! The ival should be 'ms' unit, and should consider the sampling rate!!
        end
        [a b]=find(idx==1); for i2=1:length(b), C.right(:,:,i2)=tm_trial{b(i2)}; end; label(1, b)=1; label_bin(b)=1; % 1: right class
        [a b]=find(idx==2); for i2=1:length(b), C.left(:,:,i2)=tm_trial{b(i2)};  end; label(2, b)=1; label_bin(b)=2;% left
        [a b]=find(idx==3); for i2=1:length(b), C.foot(:,:,i2)=tm_trial{b(i2)};  end; label(3, b)=1; label_bin(b)=3;% foot
        [a b]=find(idx==4); for i2=1:length(b), C.rest(:,:,i2)=tm_trial{b(i2)};  end; label(4, b)=1; label_bin(b)=4;% rest
        
        for i=1:length(tm_trial)
            active_all(:,:,i)=tm_trial{:,i};
        end
%         active_all=cat(3, C.right, C.left, C.foot, C.rest);
        
        for i=1:length(set_num)
            [a b]=find(label_bin==set_num{i}); % find target class
            C1=active_all(:,:,b);
%             C1=tm_trial{:,b};
            tm_b=[];
            for j=1: length(set_num{i,2}) % find other classes
                [a b2]=find(label_bin==set_num{i,2}{j});
                tm_b=[tm_b b2];
            end
            C2=active_all(:,:,tm_b);
            tm_label(1,1:length(b))=1;
            tm_label(2,length(b)+1:length(b)+length(tm_b))=1;
            SMT.x=cat(2, permute(C1,[1 3 2]), permute(C2, [1 3 2]));
            SMT.class={'class1', 'classe2'};
            SMT.y_logic=tm_label;
            SMT
            % CSP and LDA
            [SMT, CSP{i}, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
            FT=func_featureExtraction(SMT, {'feature','logvar'});
            [CLY_LDA{i}]=func_train(FT,{'classifier','LDA'});
            tm_label=[];
        end
        
        % Trainig done! sending it to matlab1.
        fwrite(d,1,'double');
        
        pause(0.05);
        feedback_on=true;
    end
    %%
    orig_Dat=[orig_Dat; data];
    if length(orig_Dat)>buffer_size % prevent overflow
        Dat=orig_Dat(end-buffer_size+1:end,:);
        orig_Dat=Dat;
    end
    
    if feedback_on
        Dat2.x=Dat;
        Dat2.fs=state.fs;
        %         Dat=prep_resample(Dat2,500);
        Dat=Dat2.x;
        fDat=prep_filter(Dat, {'frequency', band;'fs',1000});%state.fs });
        fDat=fDat(end-data_size:end,:); % data
        
        if iscell(CSP)
            for i=1:length(CSP)
                tm=func_projection(fDat, CSP{i});
                ft=func_featureExtraction(tm, {'feature','logvar'});
                [cf_out(i)]=func_predict(ft, CLY_LDA{i});
            end
            if toc>0.2                
                cf_out
                if sending_on
                    %                 while 1
                    fwrite(d,cf_out,'double');
                end
                %                 pause(0.5)
                %                 end
                tic
            end
            
        end
    end
    
end

if ~play
    pause(2);
    str='Bye'
    pause(3);
end

% [ keyIsDown, seconds, keyCode ] = KbCheck;
% if keyIsDown
%     if keyCode(escapeKey)
%         ShowCursor;
%         play=false;
%     elseif keyCode(waitKey)
%         warning('stop')
%         GetClicks(w);
%         Screen('Close',tex1);
%     else
%     end
% end


end

