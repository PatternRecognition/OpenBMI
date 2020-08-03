for su = 1:3;  % SUBJECT NUMBER

% GET THE DATA
file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/martigny/'];


file_list= strcat(file_dir, 'train_subject', int2str(su), '_raw0', ...
                  cellstr(int2str([1:3]')));
for ff= 1:length(file_list),
  S= load(file_list{ff});
  if ff==1,
    cnt= struct('x', S.X, 'clab',{S.nfo.clab}, 'fs',S.nfo.fs, ...
                'title', untex(S.nfo.name(1:end-2)));
    Y= S.Y;
  else
    cnt.x= cat(1, cnt.x, S.X);
    Y= cat(1, Y, S.Y);
  end
  cnt.T(ff)= size(S.X, 1);
end
mnt= setDisplayMontage(S.nfo.clab, 'martigny');
mnt.xpos= S.nfo.xpos;
mnt.ypos= S.nfo.ypos;
clear S

mrk= struct('fs',cnt.fs);
mrk.pos= find(diff([0; Y]))';
break_points= cumsum(cnt.T(1:end-1));
mrk.pos= unique([mrk.pos break_points+1]);
mrk.y= double([Y(mrk.pos)'==2; Y(mrk.pos)'==3; Y(mrk.pos)'==7]);
mrk.className= {'left', 'right', 'word'};
mrk.toe = transpose(cellstr(num2str(([1 2 3]*mrk.y)')));
mrk.block = ones(1,sum(mrk.pos<break_points(1)+1));

for i = 1:length(break_points)-1
  mrk.block = cat(2,mrk.block,(i+1)*ones(1,sum(mrk.pos>=break_points(i)+1 & mrk.pos<break_points(i+1)+1)));
end
mrk.block =  cat(2,mrk.block,(length(break_points)+1)*ones(1,sum(mrk.pos>=break_points(end)+1)));
mrk.indexedByEpochs = {'block'};

file_dir = [EEG_IMPORT_DIR 'bci_competition_iii/data_set_v_' char(su+64)];

saveProcessedEEG(file_dir,cnt,mrk,mnt);

end

