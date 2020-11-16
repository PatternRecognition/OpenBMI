% counting.m
%
%
% 
% author: Young-Seok Kweon
% created: 2020.11.12
%% init
clc; clear; close all;
%% load

list=dir('Dataset\*.mat');

%% counting

name=[];
for i=1:length(list)
    temp=list(i).name;
    temp=split(temp,'_');
    name(i)=str2num(temp{1});
end
%%
count(max(name))=0;
for i=1:length(list)
    count(name(i)) = count(name(i))+1;
end

for i=unique(count)
    if i==0
        continue;
    end
    
    n(i)=sum(count==i);
end

%%
dir_='F:\wsc\datasets\wsc-dataset-0.1.0.csv';

[num, txt, raw]=xlsread(dir_);


year=count;
c=count;
year(:)=0;
c(:)=0;
for i=1:length(list)
    if c(num(i,1))~=0
        year(num(i,1))=year(num(i,1))+num(i,3)-c(num(i,1));
        c(num(i,1))=num(i,3);
    else
        c(num(i,1))=num(i,3);
    end
end
year=year./count;
for i=unique(count)
    if i==0
        continue;
    end
    
    n(i)=mean(year(count==i));
end
