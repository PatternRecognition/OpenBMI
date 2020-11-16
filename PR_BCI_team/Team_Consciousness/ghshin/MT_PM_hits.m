%% Winter conference - Fig 3
clear; close all; clc;

Time = {'BN', '24'};
%% MT_textfile load
Path_M = 'H:\1. Journal\NeuroImage\SLEEP_DATA_24H\';
DirGroup_M = dir(fullfile(Path_M,'*'));
FileNamesGroup_M = {DirGroup_M.name};
FileNamesGroup_M = FileNamesGroup_M(1,3:end);
Time_M = {'BN','24'};
removal_word = [".jpg"];
%% load
% Data_BN=importdata('ks_min_20200330_BN_recall_visuo.txt');
ccnt = 0;
for n = 1:size(FileNamesGroup_M,2)
%% PM Task (Succeful)           
    % Recall
    for t = 1:size(Time,2)
        Data_recall=textscan(fopen([FileNamesGroup_M{n} '_' Time_M{t} '_recall_visuo.txt']), '%s %s %s %s %s %s %s');   
        Data_recall=[Data_recall{:}];
        Hash_Table_recall = str2double(erase(Data_recall([2:end],3),removal_word));
        Hash_Table_recall = [cellfun(@(x) sprintf('%02d',x),num2cell(Hash_Table_recall),'UniformOutput',false)...
            Data_recall([2:end],4) Data_recall([2:end],6) Data_recall([2:end],2)];
        Hash_Table_recall = sortrows(Hash_Table_recall,1);

        for i = 1:size(Hash_Table_recall,1)
            if string(Hash_Table_recall(i,2)) == 'o'
                res(i,t) = 1;
            elseif string(Hash_Table_recall(i,2)) == 'n'
                res(i,t) = 0;
            elseif isspace(string(Hash_Table_recall(i,2))) == 1 % i don't know
                res(i,t) = 2;
            end
        end

        % four possible reponse categories             
        temp=[1,0];% 1: old, 0: new
        for i=1:2
            for j=1:2
                if i == 1
                    trial{i,j,t} = find(res(1:38,t)==temp(j)); 
                elseif i == 2
                    trial{i,j,t} = find(res(39:end,t)==temp(j))+38;
                end
            end
        end
        if t == 1
            Suc = [trial{1,1,t}]; 
            Succ = sortrows(str2double(Hash_Table_recall(Suc,4)));
        end
    end
%% Picture memory hits
    label = intersect(trial{1,1,1}, trial{1,1,2}); 

    % Before nap
    Data_recall=textscan(fopen([FileNamesGroup_M{n} '_' Time_M{1} '_recall_visuo.txt']), '%s %s %s %s %s %s %s');   
    Data_recall=[Data_recall{:}];
    Hash_Table_recall = str2double(erase(Data_recall([2:end],3),removal_word));
    Hash_Table_recall = [cellfun(@(x) sprintf('%02d',x),num2cell(Hash_Table_recall),'UniformOutput',false)...
        Data_recall([2:end],4) Data_recall([2:end],6) Data_recall([2:end],2)];
    Hash_Table_recall = sortrows(Hash_Table_recall,1);    
    
    Label = sortrows(str2double(Hash_Table_recall(label,4))); 
    y = double(ismember(Succ, Label));
    y_ = nonzeros(y);
    STM_hits(n,1) = size(Succ,1);
    LTM_hits(n,1) = size(y_,1);
end

hits = [STM_hits LTM_hits];


        