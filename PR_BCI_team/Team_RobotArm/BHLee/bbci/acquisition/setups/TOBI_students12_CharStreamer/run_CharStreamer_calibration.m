%% matlab script for CharStreamer feedback
pause on;
% TODO brainrecorder stuff
send_xmlcmd_udp('init', '127.0.0.1', 12345)
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'CharStreamer','command','sendinit')

% offline runs
% standard configuration

stimutil_waitForMarker('S 94')  % !!! wait till feedback is loaded
fprintf('   Initialization complete! \n');
pause(1)

iterations = 13;
send_xmlcmd_udp('interaction-signal', 'b:calibration_mode' , 'True');
send_xmlcmd_udp('interaction-signal', 'b:online_mode' , 'False');
send_xmlcmd_udp('interaction-signal', 'b:online_simulation' , 'False');
send_xmlcmd_udp('interaction-signal', 'b:early_stopping' , 'False');
send_xmlcmd_udp('interaction-signal', 'i:iterations' , iterations);

log_file = fopen([TODAY_DIR 'matlab.log'], 'w');

% generate targets and condiition sequences
tgs = {'del', 'a', 'c', 'd', 'e', 'h', 'i', 'l', 'm', 'n', 'r', 's', 't', 'u', 'paus'};
targets = {{tgs{randperm(length(tgs))}}, {tgs{randperm(length(tgs))}}, {tgs{randperm(length(tgs))}}};

% random order of 5 out of 6 possible permutations of the 3 condiitions
condis = perms([1 2 3]);
r = randperm(6);
r = r(1:5);
condis = condis(r,:);

% save or load if already done
fname = [TODAY_DIR 'block_info.mat'];
if ~exist(fname, 'file')
    save(fname, 'condis', 'targets');
    disp('saved block_info into file!');
    list_true_num = {};
    list_sbj_counts = {};
else
    load(fname);
    disp('loaded block_info from file!');
end

%%
bvr_startrecording('impedanceDump', 'impedances', 1);
pause(1)
bvr_sendcommand('stoprecording')


%% standard measurements: eyes_open
bvr_startrecording('eyes_open', 'impedances', 0);
pause(45)
bvr_sendcommand('stoprecording')

%% standard measurements: eyes_closed
bvr_startrecording('eyes_closed', 'impedances', 0);
pause(45)
bvr_sendcommand('stoprecording')

% %% adding condi 4 (only for testing with michael)
% condis(:,4) = 4;
% targets{4} = targets{3};


%% Hauptmessung!

targets{1}
targets{2}
targets{3}
fprintf('press <ENTER> to start offline calibration runs\n\n');
pause;

for block = 1:5 % block iteration
    fprintf('\n \n \tBLOCK: %d - press <ENTER> to start\n', block);
    pause;
    fprintf(log_file, 'BLOCK %d\n', block);

    for j = 1:3 % condiition iteration
        condi = condis(block, j);

        fprintf('\tcondiition: %d - press <ENTER> to start\n', condi);
        pause;

        fprintf(log_file, '\tcondiition: %d\n', condi);
        % set condiition parameters
        switch condi
            case 1
                send_xmlcmd_udp('interaction-signal', 'i:condiition' , condi);
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 750);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'True');
            case 2
                send_xmlcmd_udp('interaction-signal', 'i:condiition' , condi);
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 250);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'True');
            case 3
                send_xmlcmd_udp('interaction-signal', 'i:condiition' , condi);
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 250);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'False');
            case 4
                send_xmlcmd_udp('interaction-signal', 'i:condiition' , condi);
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 750);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'False');

        end
        fname = ['condi' num2str(condi) '_' VP_CODE];
        bvr_startrecording(fname, 'impedances', 0);
        for i = 1:3 % target iteration
            target = targets{condi}{3 * (block-1) + i};
            send_xmlcmd_udp('interaction-signal', 's:target' , target);
            fprintf('\t\ttarget: %s\n', target);
            fprintf(log_file, '\t\ttarget: %s', target);

            pre_iterations = 1;
            send_xmlcmd_udp('interaction-signal', 'i:pre_iterations' , pre_iterations);
            fprintf(log_file, '\titerations=%d + %d\n', iterations, pre_iterations);

            send_xmlcmd_udp('interaction-signal', 'command', 'play')
            %stimutil_waitForMarker(92) % könnte genutzt werden um
            %richtiges trial end von manuellem stop zu unterscheiden und
            %jeweils anders zu behandeln
            fprintf('Trial is running...');
            stimutil_waitForMarker('S 92')
            fprintf('   Trial finished\n');

            pause
                        sbj_counts = '';
                        while ~isnumeric(sbj_counts)
                            %catch counts of the trials!!
                            sbj_counts = input('enter the RATING: ', 's');
                            sbj_counts = str2num(sbj_counts);
                            if ~isempty(sbj_counts)
                                list_sbj_counts{condi}(((block-1)*3)+i) = sbj_counts;
                            else
                                sbj_counts = '';
                            end
                        end
%                         true_num = '';
%                         while ~isnumeric(true_num)
%                             true_num = input('enter the TRUE number of Targets: ', 's');
%                             true_num = str2num(true_num);
%                             if ~isempty(true_num)
%                                 list_true_num{condi}(((block-1)*3)+i) = true_num;
%                             else
%                                 true_num = '';
%                             end
%                         end
        end
        bvr_sendcommand('stoprecording')

    end
    save([TODAY_DIR 'counts.mat'], 'list_true_num', 'list_sbj_counts');
    fprintf('saved counts into file')
end

fprintf('press <ENTER> to stop feedback and close evertything!')
pause()
% stop
send_xmlcmd_udp('interaction-signal', 'command', 'stop');

% quit
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

fclose(log_file);




%% metal notes, ToDos for Konrad

% V and paus too similar
% del hat knacker
% zwischen Z und les knacker
% d und del ist ein problem
% H und K - michale hat ideen
% D G E C --> E schwingt durch....  G geht davon noch am besten



%% remaining targets for Michael

tgs = {'del', 'a', 'c', 'd', 'e', 'h', 'i', 'l', 'm', 'n', 'r', 's', 't', 'u', 'paus'};
tgs2 = {'b', 'f', 'g', 'j', 'k', 'o', 'p', 'q', 'v', 'w', 'x', 'y', 'z', 'leer', 'les'};

rand_target_list = tgs2(randperm(length(tgs2)));

for ii = 1:length(rand_target_list)
    for condi = 2:3
        switch condi
            case 2
                send_xmlcmd_udp('interaction-signal', 'i:condiition' , condi);
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 250);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'True');
            case 3
                send_xmlcmd_udp('interaction-signal', 'i:condiition' , condi);
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 250);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'False');
        end
        send_xmlcmd_udp('interaction-signal', 's:target' , rand_target_list{ii});

        fname = ['condi' num2str(condi) '_' VP_CODE];


        pre_iterations = 1;
        send_xmlcmd_udp('interaction-signal', 'i:pre_iterations' , pre_iterations);
        bvr_startrecording(fname, 'impedances', 0);
        pause(1)
        send_xmlcmd_udp('interaction-signal', 'command', 'play')
        %stimutil_waitForMarker(92) % könnte genutzt werden um
        %richtiges trial end von manuellem stop zu unterscheiden und
        %jeweils anders zu behandeln
        fprintf('Trial is running...');
        stimutil_waitForMarker('S 92')
        fprintf('   Trial finished\n');
        bvr_sendcommand('stoprecording')
        
        pause

    end
end