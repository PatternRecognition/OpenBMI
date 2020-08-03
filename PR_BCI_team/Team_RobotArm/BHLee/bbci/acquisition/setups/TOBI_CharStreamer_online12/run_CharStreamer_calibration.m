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
tgs = {'leer', 'paus', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'les', 'del'};
allTargets = [ tgs tgs] ; %for 2 conditions
rand_order =  randperm(length(tgs)*2);
isclass2 = rand_order > 30;    
    
targets = {allTargets{rand_order}};

% save or load if already done
fname = [TODAY_DIR 'block_info.mat'];
if ~exist(fname, 'file')
    save(fname, 'targets', 'isclass2', 'rand_order');
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



fprintf('press <ENTER> to start offline calibration runs\n\n');
pause;

for block = 1:length(targets) % block iteration
    fprintf('\n \n \tBLOCK: %d - press <ENTER> to start\n', block);
    pause;
    fprintf(log_file, 'BLOCK %d\n', block);

    for j = 1:6 % condiition iteration
        condi = condis(block, j);

        fprintf('\tcondiition: %d - press <ENTER> to start\n', condi);
        pause;

        fprintf(log_file, '\tcondiition: %d\n', condi);
        % set condiition parameters
        switch isclass2(itrial)
            case 0
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 250);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'True');
                condi_description = 'SOA_250_randomOrder_';
            case 1
                send_xmlcmd_udp('interaction-signal', 'i:SOA' , 250);
                send_xmlcmd_udp('interaction-signal', 'b:random' , 'False');
                condi_description = 'SOA_250_fixOrder_';

        end
        fname = ['CharStreamer_condi' condi_description VP_CODE];
        bvr_startrecording(fname, 'impedances', 0);
        for i = 1:5 % target iteration
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




%%
% my_rand_order =  randperm(30);
% my_targets = tgs(my_rand_order);
% fname = ['CharStreamer_condi' condi_description VP_CODE];

% bvr_startrecording(fname, 'impedances', 0);
for zz = 2:30 % target iteration
            target = my_targets{zz};
            send_xmlcmd_udp('interaction-signal', 's:target' , target);
            fprintf('\t\ttarget: %s\n', target);
            fprintf(log_file, '\t\ttarget: %s', target);

            pre_iterations = 0;
            send_xmlcmd_udp('interaction-signal', 'i:pre_iterations' , 0);
            fprintf(log_file, '\titerations=%d + %d\n', iterations, 0);
            
            my_iterations = 16;
            send_xmlcmd_udp('interaction-signal', 'i:iterations' , my_iterations);

            send_xmlcmd_udp('interaction-signal', 'command', 'play')
            %stimutil_waitForMarker(92) % könnte genutzt werden um
            %richtiges trial end von manuellem stop zu unterscheiden und
            %jeweils anders zu behandeln
            fprintf('Trial is running...');
            stimutil_waitForMarker('S 92')
            fprintf('   Trial finished\n');

                        sbj_counts = '';
                        while ~isnumeric(sbj_counts)
                            %catch counts of the trials!!
                            sbj_counts = input('enter the RATING: ', 's');
                            sbj_counts = str2num(sbj_counts);
%                             if ~isempty(sbj_counts)
%                                 list_sbj_counts{condi}(((block-1)*3)+i) = sbj_counts;
%                             else
%                                 sbj_counts = '';
%                             end
                        end

end