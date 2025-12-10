% labeled matrix image

N=10;
P=10;
M = rand(N,P);
rowIDs = 1:N * 10;
colIDs = (1:P +3) * 5;
imagesc(M);
% colormap(parula);
colorbar;

xticks(1:length(colIDs));
yticks(1:length(rowIDs));

xticklabels(colIDs);
yticklabels(rowIDs);

% Make the tick-label font smaller
ax = gca;
ax.FontSize = 8;     % or smaller (e.g., 5, 4)

xtickangle(90);      % optional: rotate to fit more labels vertically

xlabel('Column ID');
ylabel('Row ID');

hold on;
x1 = 0; x2 = 10.5;
y1 = 4.5; y2 = 4.5;
plot([x1,x2],[y1,y2],'Color','r','LineWidth',4);
