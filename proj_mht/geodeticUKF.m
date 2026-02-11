% UKF: Geodetic Position (Lat, Lon, H) and ENU Velocity
clear; clc;

%% 1. Constants & Setup
a = 6378137.0;              % WGS84 Semi-major axis (m)
f = 1/298.257223563;        % Flattening
e2 = 2*f - f^2;             % eccentricity squared

dt = 1.0;                   % Time step
T_geo = [0.5934, -1.239, 10]; % Transmitter [Lat, Lon, H] (rad)
R_geo = [0.5935, -1.238, 10]; % Receiver [Lat, Lon, H] (rad)

% State: [lat(rad), lon(rad), h(m), ve(m/s), vn(m/s), vu(m/s)]
x = [0.59345; -1.2385; 500; 20; 10; 0]; 
P = diag([1e-8, 1e-8, 10, 5, 5, 2]); 
Q = diag([1e-12, 1e-12, 0.1, 0.5, 0.5, 0.1]);
R = diag([20^2, deg2rad(0.5)^2, 1^2]); 

% UKF Weights
n = 6; alpha = 1e-3; ki = 0; beta = 2;
lambda = alpha^2 * (n + ki) - n;
wm = [lambda/(n+lambda), repmat(1/(2*(n+lambda)), 1, 2*n)];
wc = wm; wc(1) = wc(1) + (1 - alpha^2 + beta);

%% 2. Prediction Step (Geodetic Kinematics)
A = chol(P + Q)';
sigmas = [x, x + sqrt(n+lambda)*A, x - sqrt(n+lambda)*A];

for i = 1:(2*n+1)
    lat = sigmas(1,i); h = sigmas(3,i);
    ve = sigmas(4,i); vn = sigmas(5,i); vu = sigmas(6,i);
    
    % Radii of Curvature
    M = a*(1-e2) / (1 - e2*sin(lat)^2)^1.5;
    N = a / sqrt(1 - e2*sin(lat)^2);
    
    % Derivatives: dLat = Vn/(M+h), dLon = Ve/((N+h)cos(Lat))
    sigmas(1,i) = lat + (vn / (M + h)) * dt;
    sigmas(2,i) = sigmas(2,i) + (ve / ((N + h) * cos(lat))) * dt;
    sigmas(3,i) = h + vu * dt;
    % Velocities (Constant Velocity model in ENU)
    % Note: In high-precision long-range, you'd add Coriolis/Transport rates here
end

x_pred = sum(wm .* sigmas, 2);
P_pred = Q;
for i = 1:2*n+1
    d = sigmas(:,i) - x_pred;
    P_pred = P_pred + wc(i) * (d * d');
end

%% 3. Measurement Step (Bistatic in Geodetic)
Z_sig = zeros(3, 2*n+1);
for i = 1:(2*n+1)
    % Convert Sigma Point (LLA) to ECEF to calculate Bistatic Range
    pos_ecef = lla2ecef_local(sigmas(1:3,i), a, e2);
    T_ecef   = lla2ecef_local(T_geo, a, e2);
    R_ecef   = lla2ecef_local(R_geo, a, e2);
    
    Rt_vec = pos_ecef - T_ecef;
    Rr_vec = pos_ecef - R_ecef;
    Rt = norm(Rt_vec);
    Rr = norm(Rr_vec);
    
    % Bistatic Range
    Z_sig(1,i) = Rt + Rr;
    
    % Angle of Transmission (Azimuth in T's ENU frame)
    % Requires converting target pos to Transmitter-centered ENU
    enu_target = ecef2enu_local(pos_ecef, T_geo, a, e2);
    Z_sig(2,i) = atan2(enu_target(1), enu_target(2));
    
    % Range Rate (Doppler)
    % Convert ENU velocity to ECEF for vector dot product
    v_enu = sigmas(4:6,i);
    v_ecef = enu2ecef_vel(v_enu, sigmas(1:2,i));
    Z_sig(3,i) = dot(Rt_vec/Rt, v_ecef) + dot(Rr_vec/Rr, v_ecef);
end

% Standard UKF Update (Mean/Covariance/Gain)
z_hat = sum(wm .* Z_sig, 2);
S = R; Pxz = zeros(n, 3);
for i = 1:2*n+1
    dz = Z_sig(:,i) - z_hat;
    dx = sigmas(:,i) - x_pred;
    S = S + wc(i)*(dz*dz');
    Pxz = Pxz + wc(i)*(dx*dz');
end
K = Pxz / S;
x = x_pred + K * ([2550; 0.66; -10] - z_hat); % Example observation
P = P_pred - K * S * K';

%% Local Helper Functions
function ecef = lla2ecef_local(lla, a, e2)
    lat = lla(1); lon = lla(2); h = lla(3);
    N = a / sqrt(1 - e2*sin(lat)^2);
    ecef = [(N+h)*cos(lat)*cos(lon); (N+h)*cos(lat)*sin(lon); (N*(1-e2)+h)*sin(lat)];
end

function enu = ecef2enu_local(ecef, ref_lla, a, e2)
    ref_ecef = lla2ecef_local(ref_lla, a, e2);
    rel = ecef - ref_ecef;
    lat = ref_lla(1); lon = ref_lla(2);
    R = [-sin(lon) cos(lon) 0; -sin(lat)*cos(lon) -sin(lat)*sin(lon) cos(lat); cos(lat)*cos(lon) cos(lat)*sin(lon) sin(lat)];
    enu = R * rel;
end

function v_ecef = enu2ecef_vel(v_enu, lla)
    lat = lla(1); lon = lla(2);
    R = [-sin(lon) -sin(lat)*cos(lon) cos(lat)*cos(lon); ...
          cos(lon) -sin(lat)*sin(lon) cos(lat)*sin(lon); ...
          0         cos(lat)           sin(lat)];
    v_ecef = R * v_enu;
end
