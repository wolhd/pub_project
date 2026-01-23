% --- 1. Setup Parameters ---
dt = 0.5; % Time step

% Define the 1D building block
% [pos_next] = [1  dt] * [pos]
% [vel_next]   [0   1]   [vel]
F_1d = [1 dt; 
        0  1];

% Expand to 3D using Kronecker product (creates a 6x6 block diagonal matrix)
F = kron(eye(3), F_1d);

% Process Noise Covariance (6x6)
% We'll assume a small uncertainty in the constant velocity assumption
Q_1d = [0.01 0; 
        0    0.01];
Q = kron(eye(3), Q_1d);

% --- 2. Initial Values ---
% Initial State: [x; vx; y; vy; z; vz]
% Starting at origin (0,0,0) with velocity (2, 1, 0.5) m/s
x_est = [0; 2;   % X-axis
         0; 1;   % Y-axis
         0; 0.5]; % Z-axis

% Initial Uncertainty (6x6 Identity matrix)
P_est = eye(6);

% --- 3. THE PREDICT STEP ---
x_pred = F * x_est;             % Predict next state
P_pred = F * P_est * F' + Q;    % Predict uncertainty

% --- Display Results ---
fprintf('Predicted 3D Position: [%.2f, %.2f, %.2f]\n', ...
    x_pred(1), x_pred(3), x_pred(5));
fprintf('Predicted 3D Velocity: [%.2f, %.2f, %.2f]\n', ...
    x_pred(2), x_pred(4), x_pred(6));
