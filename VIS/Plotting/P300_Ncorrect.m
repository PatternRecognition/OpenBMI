function Accuracy=P300_Ncorrect(smt,filepath,subject_info,spellerText_on)

file=fullfile(filepath,subject_info.subject,subject_info.session, 'log');
% get log file
list=ls(sprintf('%s\\*_p300_param*', file));
load_file=fullfile(file,list(end,:));
load(load_file);

score=0;
for i=1:length(spellerText_on)
    if strcmp(SAVE{i}.char(1),spellerText_on(i))
        score=score+1;
    end
end

Accuracy=score/length(spellerText_on);
disp(Accuracy);

for i=1:length(spellerText_on)
    a(i)=SAVE{i}.char(1);
    disp(SAVE{i}.char(1));
end
end