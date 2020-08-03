files = {'260_english.txt'};
a=textread(files{1},'%s','delimiter','\n');
k=1;
clear inx words

for i=1:260
    while strcmp(a{i}(end),' ')
        a{i}=a{i}(1:end-1);
    end
end

for i=1:260
    for j=1:260
        %         if i==32 && j==229
        %             a{i}
        %             a{j}
        %             strcmp(a{i},a{j})
        %         end

        if strcmp(a{i},a{j}) && i~=j% && i<j
            inx(k)= i;
            words{k}=a{i};
            k=k+1;
        end
    end
end
inx
words
