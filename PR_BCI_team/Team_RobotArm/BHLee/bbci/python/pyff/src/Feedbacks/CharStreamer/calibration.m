% matlab script for CharStreamer feedback
pause on;
% TODO brainrecorder stuff

send_xmlcmd_udp('init', '127.0.0.1', 12345)

send_xmlcmd_udp('interaction-signal', 's:_feedback', 'CharStreamer','command','sendinit')

%% !!! wait till feedback is loaded

%% offline runs
% standard configuration
iterations = 5;
send_xmlcmd_udp('interaction-signal', 'b:calibration_mode' , 'True');
send_xmlcmd_udp('interaction-signal', 'b:online_mode' , 'False');
send_xmlcmd_udp('interaction-signal', 'b:online_simulation' , 'False');
send_xmlcmd_udp('interaction-signal', 'i:iterations' , iterations);



log_file = fopen([TODAY_DIR '\matlab.log'], 'w');

% generate targets and condition sequences
tgs = {'a', 'e', 'i', 'o', 'u', 'del', 'leer', 'c', 'f', 't', 'b', 'm', 'n', 'p', 'd'};
targets = {{tgs{randperm(length(tgs))}}, {tgs{randperm(length(tgs))}}, {tgs{randperm(length(tgs))}}};

% random order of 5 out of 6 possible permutations of the 3 conditions
conds = perms([1 2 3]);
r = randperm(6);
r = r(1:5);
conds = conds(r,:);

% save or load if already done
fname = [TODAY_DIR 'block_info.mat'];
if ~exist(fname, 'file')
   save(fname, 'conds', 'targets');
   disp('saved block_info into file!');
else 
   load(fname);
   disp('loaded block_info from file!');
end

%%
fprintf('press <ENTER> to start offline runs\n\n');
pause;

targets{1}
targets{2}
targets{3}

for block = 1:5 % block iteration
    fprintf('BLOCK %d\n', block);
    fprintf(log_file, 'BLOCK %d\n', block);

    for j = 1:3 % condition iteration
        cond = conds(block, j);
        
        fprintf('\tcondition: %d - press <ENTER>\n', cond);
        fprintf(log_file, '\tcondition: %d\n', cond);
        pause;
        % set condition parameters
        switch cond
            case 1
                send_xmlcmd_udp('interaction-signal', 'i:condition' , cond);
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 750);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'False');
            case 2
                send_xmlcmd_udp('interaction-signal', 'i:condition' , cond);
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 250);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'True');
            case 3
                send_xmlcmd_udp('interaction-signal', 'i:condition' , cond);
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 250);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'False');
        end   
                
        bvr_startrecording(fname);
        for i = 1:3 % target iteration        
            target = targets{cond}{3 * (block-1) + i};
            send_xmlcmd_udp('interaction-signal', 's:target' , target);
            fprintf('\t\ttarget: %s\n', target);
            fprintf(log_file, '\t\ttarget: %s', target);
            
            pre_iterations = round(rand) + 1;
            send_xmlcmd_udp('interaction-signal', 'i:pre_iterations' , pre_iterations);
            fprintf(log_file, '\titerations=%d + %d\n', iterations, pre_iterations);
            
            send_xmlcmd_udp('interaction-signal', 'command', 'play')
            
            %stimutil_waitForMarker(91) % könnte genutzt werden um
            %richtiges trial end von manuellem stop zu unterscheiden und
            %jeweils anders zu behandeln
            

            
            true_num = '';
            while isnumeric(true_num)
                true_num = input('enter the TRUE number of Targets', 's');
                    true_num = str2num(true_num);
                    if ~isempty(true_num)
                        list_true_num{cond}(((block-1)*3)+i) = true_num;
                    end
                end
            end
            sbj_counts = '';
            while ~isnumeric(sbj_counts)
                %catch counts of the trials!!
                sbj_counts = input('enter the counted number (when block completed)', 's');
                    sbj_counts = str2num(sbj_counts);
                    if ~isempty(sbj_counts)
                        list_sbj_counts{cond}(((block-1)*3)+i) = sbj_counts;
                    end
                end
            end

            

        end
        %bvr_sendcommand('stoprecording')

    end
end


%% stop
send_xmlcmd_udp('interaction-signal', 'command', 'stop');

% quit
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

fclose(log_file);