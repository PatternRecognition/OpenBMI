% converting_edf.m
%
% converting edf data to mat
%
% 
%
% author: Young-Seok Kweon
% created: 2020.11.11
%% init
clc; close all; clear;
edf_dir='F:\edf';
addpath(edf_dir);
data_dir='C:\Users\�ǿ뼮\wsc\polysomnography\';
%% load
filename = dir([data_dir,'*.edf']);

for i=1:length(filename)
    name=filename(i).name;
    [hdr, record]=edfread([data_dir,name]);
    [ch,tp]=size(record);
    n_trials=floor(tp/3000);
    temp=[];
    for j=1:n_trials
        temp(:,:,j)=record(:,(j-1)*3000+1:j*3000);
    end
    record=temp;
    temp=split(name,'.');
    anno=fileread([data_dir,temp{1},'.eannot']);
    anno=split(anno);
    chname=hdr.label;
    list=split(name,'-');
    save_name=[list{3},'_',list{2}(end)];
    save(['Dataset\',save_name],'record','chname','anno');
    fprintf('%s Done!\n',name);
end
