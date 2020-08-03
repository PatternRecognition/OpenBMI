function mrk = trennfinger(mrk)
%TRENNFINGER separates finger
%
% usage:
%    mrk = trennfinger(mrk);
%
% input:
%    mrk      a usual mrk structure for selfpaced
% 
% output:
%    mrk      a mrk structure where each finger is a class
%
% Guido DOrnhege, Volker Kunzmann 04/07/2003


sym = [  65    83    68    70    74    75    76   192];

names = {'left V','left IV','left III','left II','right II','right III','right IV','right V'};

fi = unique(abs(mrk.toe));
% (kraulem): prevents crashing if mrk.toe contains additional markers:
fi = intersect(fi,sym);

mrk.y = zeros(length(fi),size(mrk.y,2));
mrk.className = cell(1,length(fi));

for i = 1:length(fi)
    j = find(fi(i)==sym);
    ind = find(abs(mrk.toe)==fi(i));
    mrk.y(i,ind) = 1;
    mrk.className{i} = names{j};
end
