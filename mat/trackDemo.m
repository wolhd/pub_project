clc; clear;

% PARAMETERS
numSteps = 50;       % Number of time steps
numTargets = 2;      % Number of targets
dt = 1.0;            % Time step duration
posNoise = 1.0;      % Position noise standard deviation

% INITIALIZE TRUE STATES (x, y, vx, vy)
states = [0,  0, 1, 0.5;     % Target 1
          20, 10, -0.5, -1]; % Target 2

% CREATE KALMAN FILTERS FOR EACH TARGET
filters = cell(1, numTargets);
tracks = zeros(numSteps, 4, numTargets);

for i = 1:numTargets
    filters{i} = trackingKF('MotionModel','2D Constant Velocity',...
                            'MeasurementModel', [1 0 0 0; 0 1 0 0],...
                            'MeasurementNoise', posNoise^2 * eye(2));
    filters{i}.State = states(i, :)';
end

% SIMULATION LOOP
for t = 1:numSteps
    figure(1); clf; hold on;
    title(['Time step: ', num2str(t)]);
    axis([-10 30 -10 30]); grid on;

    for i = 1:numTargets
        % Simulate true motion
        states(i, 1:2) = states(i, 1:2) + states(i, 3:4) * dt;

        % Generate detection (with noise)
        detection = states(i, 1:2) + posNoise * randn(1,2);

        % Correct and predict using Kalman filter
        correct(filters{i}, detection');
        predict(filters{i});

        % Store the track for plotting
        tracks(t, :, i) = filters{i}.State';

        % Plot
        plot(states(i,1), states(i,2), 'go', 'MarkerSize', 10, 'DisplayName','True Position');
        plot(detection(1), detection(2), 'rx', 'MarkerSize', 10, 'DisplayName','Detection');
        plot(filters{i}.State(1), filters{i}.State(2), 'b*', 'MarkerSize', 10, 'DisplayName','Estimated');
    end

    legend('Location','northwest');
    pause(0.1);
end
