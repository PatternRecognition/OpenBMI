year= 2003:2011;
%% Retrieved Nov 29th
citations= [1 5  4  4  3  3  2  3 1;
            0 7 19 20 23 30 23 14 12;
            0 0  0  1 14 11 24 18 13];
colOrder= [1 0.85 0; 1 0 0; 0.75 0 1];

clf;
h= bar(citations');
set(gca, 'XTickLabel',cprintf('%d',year), 'YGrid',1);
xlabel('Year');
ylabel('Number of citations');
legend('I', 'II', 'III', 'location','NorthWest');
for cc= 1:size(colOrder,1),
  set(h(cc), 'FaceColor',colOrder(cc,:));
end

printFigure('/tmp/hist_citations', [12 6], 'format','pdf');


citations= [0 1  6 22 19 18 18 17 13; ...
            0 9 27 58 77 61 65 51 40 ...
           ];
clf;
h= bar(citations');
set(gca, 'XTickLabel',cprintf('%d',year), 'YGrid',1);
xlabel('Year');
ylabel('Number of citations');
legend('I', 'II', 'location','NorthWest');
for cc= 1:length(h),
  set(h(cc), 'FaceColor',colOrder(cc,:));
end

printFigure('/tmp/hist_citations_of_winners_I+II', [12 6], 'format','pdf');


year= 2003:2011;
citations= [0 9 27 58 77 61 65 51 40]
clf;
h= bar(citations');
xtick= cprintf('%d',year);
xtick{1}= '';
set(gca, 'XTickLabel',xtick, 'YGrid','on');
xlabel('Year');
ylabel('Number of citations');
legend('II', 'location','NorthWest');
set(h, 'FaceColor',colOrder(2,:));

opt_fig= struct('format','pdf', ...
                'folder', [TEX_DIR 'presentation/bci_competition_iv/publication/overview_article/']);
printFigure('hist_citations_of_winners_II', [12 6], opt_fig);
