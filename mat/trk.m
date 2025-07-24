
%% PARAMETERS
numSteps = 50;       % Number of steps in the random walk
stepSize = 1;         % Size of each step
numTracks = 2;
numDetsPerTP = 4;
detSpacing = 4;
xs = [];
ys = [];
dxs = [];
dys = [];
for i = 1:numTracks
    [x,y] = genTrack(numSteps, stepSize);
    xs(:,end+1) = x;
    ys(:,end+1) = y;
    
    [dx,dy] = genDets(x',y',numDetsPerTP,detSpacing);
    dxs(:, end+1:end+numDetsPerTP) = dx;
    dys(:, end+1:end+numDetsPerTP) = dy;
end
%% PLOT
figure;
plot(xs, ys, 'x-', 'LineWidth', 2); hold on;
plot(xs(1,:), ys(1,:), 'go', 'MarkerSize', 10, 'DisplayName','Start');
plot(xs(end,:), ys(end,:), 'ro', 'MarkerSize', 10, 'DisplayName','End');
grid on;
axis equal;
title('2D Random Walk');
xlabel('X'); ylabel('Y');
legend('Path', 'Start', 'End');
figure;
plot(dxs(:),dys(:), 'x');
function [dxs, dys] = genDets(x, y, numDetsPerTP, spacing)
    arrx = (rand(length(x), numDetsPerTP)-.5)*2 * spacing;
    arry = (rand(length(y), numDetsPerTP)-.5)*2 * spacing;
    dxs = x + arrx;
    dys = y + arry;
end

function [x,y,ds] = genTrack(numSteps, stepSize)

    %% INITIALIZATION
    x = zeros(1, numSteps);
    y = zeros(1, numSteps);
    x(1) = rand() * 40 - 10;
    y(1) = rand() * 40 - 10;
    % Random walk loop
    prevAngle = 2*pi*rand();
    for t = 2:numSteps
        angle = prevAngle +  pi/3 * (rand()-.5); % Random direction
        prevAngle = angle;
        dx = stepSize * cos(angle);
        dy = stepSize * sin(angle);

        x(t) = x(t-1) + dx;
        y(t) = y(t-1) + dy;
    end
end
%%
%google matlab calculate overlapping spans
