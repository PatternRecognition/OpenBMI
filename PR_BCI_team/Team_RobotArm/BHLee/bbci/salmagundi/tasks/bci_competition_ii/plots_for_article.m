name_list= {'zhang', 'neal', 'torre', 'saffari', 'chung', 'huang', ...
            'roerig','brugger', 'schroeder', 'rissacher', 'smith', ...
            'hoffmann', 'delriovera', 'mensh', 'mbwana'};

sub_dir= [EEG_IMPORT_DIR 'bci_competition_ii_submissions/'];

cd([BCI_DIR 'tasks/bci_competition_ii']);
load('our_export');
y_truth= [0 1]*mrk.y(:,test_idx);

nSubmissions= length(name_list);
YY= zeros(length(y_truth), nSubmissions);
EE= zeros(size(YY);
for ii= 1:nSubmissions,
  file= [sub_dir name_list{ii} '_result'];
  if exist([file '.mat'], 'file'),
    load([file '.mat']);
  else
    y_test= load([file '.txt']);
  end
  if any(y_test<0),
    y_test(find(y_test<0))= 0;
  end
  err(ii)= sum(y_test(:)~=y_truth(:));
  YY(:,ii)= y_test(:);
  EE(:,ii)= (y_test(:)~=y_truth(:));
end

[so,si]= sort(err);
ns= 8;

ee= EE(:,si(1:ns));
[so,ti]= sort(sum(ee,2));
colormap(linspace(1,0,10)'*[1 1 1]);
imagesc(ee(ti,:)');
set(gca, 'yTick',1:ns);
%xlabel('[trials]');
set(gca, 'xTick',[0.5 20:20:100], 'xTickLabel','[trials]|20|40|60|80|100')
ylabel('[submissions]');
shiftAxesUp;
saveFigure_png('compet_cover/berlin_errors', [12 5]);

cc= abs(corrcoef(ee));
%imagesc(cc);
%colorbar;

%ei= [4 5 1 3 6 2 8 7];
ei= [5 1 3 6 2 8 7 4];
imagesc(cc(ei,ei));
axis square;
ax= gca;
set(ax, 'xTick',1:ns, 'yTick',1:ns, ...
        'xTickLabel',ei, 'yTickLabel',ei, 'xAxisLocation','top', ...
        'tickLength',[0 0], 'cLim',[0 1]);
%pos= get(ax, 'position');
hc= colorbar;
axes(hc);
ylabel('[absolute correlation coefficients]');
%col_pos= get(hc, 'position');
%col_pos([2 4])= pos([2 4]);
%set(hc, 'position',col_pos);
axes(ax);
shiftAxesLeft;
saveFigure_png('compet_cover/berlin_correlations', [12 9]*0.7);

fv= struct('x',YY', 'y',[y_truth; 1-y_truth]);
%fv= struct('x',YY(:,si(1:8))', 'y',[y_truth; 1-y_truth]); %% this is worse
doXvalidation(fv, 'LDA', [1 1]);
